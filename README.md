# Story - A Hierarchical LLM Framework for Automated Long-Form Narrative Synthesis with Stylistic Control
My Most Worked On AI Yet

## Files that work well:
- main.py
- supercomputer_wont_be_enough_main.py
- novel_generator.py
- ULTIMATE_POWER.py
- I_blew_up_a_super_computer.py
- 100.py

**Version: 2.0 (app_v2.py)**

This project leverages Large Language Models (LLMs) via Ollama and the Langchain framework to automatically generate a complete novel. It takes a PDF resume (to inspire the main character), a story subject/premise, a target author's writing style, and a genre as input, and outputs a `.docx` file containing the generated novel.

The system is designed as a sequential pipeline of specialized "chains," each responsible for generating a specific component of the novel, from character profiles to detailed plot outlines and finally, the narrative content itself.

## Features

* **Resume-based Character Generation:** Creates a rich character profile by analyzing a provided PDF resume.
* **Contextual Setting & Theme Development:** Generates atmospheric settings and core literary themes based on the character, subject, and genre.
* **Dynamic Title Generation:** Suggests a compelling novel title.
* **Detailed Plot Outlining:** Constructs a multi-act plot structure with key turning points.
* **Chapter & Event Breakdown:** Divides the plot into chapters with summaries, and further breaks down chapters into specific plot events.
* **Narrative Generation:** Writes narrative paragraphs for each event, emulating a specified author's style.
* **"Show, Don't Tell" Focus:** The writer chain is prompted to use descriptive language, actions, and internal monologue.
* **Continuity Management:** Passes context (previous events, previously written paragraphs in a chapter) to the writer chain to maintain narrative flow.
* **Optional Refinement Pass:** An experimental chain to polish the generated prose for style, flow, and thematic resonance.
* **Configurable LLM:** Easily switch between different Ollama models.
* **`.docx` Output:** Produces a formatted document with front matter (title, genre, themes, setting, chapter list) and the full novel content.
* **Robust Error Handling:** Designed to log errors and attempt to continue generation where possible, providing informative messages.

## Architecture & Workflow

The system operates as a pipeline, where the output of one stage serves as input for subsequent stages. Each stage is managed by a dedicated Langchain `LLMChain`.

```
+--------------------------------------+
| Inputs:                              |
| - PDF Resume (e.g., divi_1.pdf)      |
| - Story Subject/Premise              |
| - Author's Style (e.g., Ocean Vuong) |
| - Genre (e.g., Psychological Fic.)   |
+--------------------------------------+
           |
           V
+--------------------------------------+
| Ollama + LLM (e.g., gemma2, llama3)  |
| (Locally Hosted Language Model)      |
+--------------------------------------+
           |
           V
+--------------------------------------+
|   app_v2.py (Orchestration Script)   |
|  - Initializes LLM via create_llm()  |
|  - Manages Chain Execution          |
+--------------------------------------+
           |
           +------> [1. MainCharacterChain] ---> Character Profile (from Resume)
           |
           +------> [2. SettingChain] ---------> Setting Description
           |          (Uses: Subject, Genre, Profile)
           |
           +------> [3. ThemeChain] -----------> Core Themes
           |          (Uses: Subject, Genre, Profile, Setting)
           |
           +------> [4. TitleChain] -----------> Novel Title
           |          (Uses: Subject, Genre, Author, Profile, Setting, Themes)
           |
           +------> [5. PlotChain] ------------> Detailed Plot Outline
           |          (Uses: Subject, Genre, Author, Profile, Title, Setting, Themes)
           |          (Includes a helper chain for dynamic plot features)
           |
           +------> [6. ChaptersChain] --------> Chapter List & Summaries
           |          (Uses: Subject, Genre, Author, Profile, Title, Plot, Setting, Themes)
           |
           +------> [7. EventChain] -----------> Events per Chapter (List)
           |          (Loop for each chapter)
           |          (Uses: Plot, Profile, Themes, Chapter Title & Summary, Author)
           |
           +------> [8. WriterChain] ----------> Narrative for each event
           |          (Loop for each chapter & each event within the chapter)
           |          (Uses: Genre, Author, Title, Profile, Plot, Setting, Themes,
           |           Chapter Info, Previous Events History, Previous Paragraphs, Current Event)
           |
           +------> [9. RefinementChain] ------> (Optional) Polished Chapter Text
           |          (Loop for each fully written chapter)
           |
           V
+--------------------------------------+
|   DocWriter Class                    |
|  - Assembles Front Matter & Content  |
+--------------------------------------+
           |
           V
+--------------------------------------+
| Output:                              |
| - [Sanitized_Title].docx             |
+--------------------------------------+
```

## Technical Details

### Key Components:

1.  **`create_llm(temperature, top_p, top_k)`:**
    * **Purpose:** Initializes and configures the `OllamaLLM` instance.
    * **Technicality:** Allows setting `temperature` (randomness), `top_p` (nucleus sampling), and `top_k` (top-k sampling) for the LLM, enabling fine-tuned control over generation creativity vs. coherence for different tasks. It also sets the Ollama base URL, model name, and request timeout. Includes a basic connection test.

2.  **Chain Classes (e.g., `MainCharacterChain`, `SettingChain`, etc.):**
    * Each chain inherits no specific base class but follows a pattern:
        * **`PROMPT` (Constant):** A carefully crafted string template using Langchain's `PromptTemplate` format. These prompts guide the LLM to generate specific information based on the provided context variables.
        * **`__init__(self)`:** Initializes its own `OllamaLLM` instance (via `create_llm` with potentially different parameters suited for its task) and an `LLMChain` from Langchain.
        * **`run(self, ...inputs)`:** Takes necessary inputs, invokes the `LLMChain` with these inputs, and processes the LLM's response.
        * **`parse_...(self, response)` (in some chains):** Custom methods (e.g., `parse_themes`, `parse_chapters`, `parse_events`) to structure the raw text output from the LLM into a more usable format (dictionaries, lists) using regular expressions (`re`) and string manipulation.

3.  **Specific Chain Functionalities:**
    * **`MainCharacterChain`:** Analyzes resume text (loaded via `PyPDFLoader`) to infer character traits, motivations, flaws, and narrative potential.
    * **`SettingChain`:** Generates key locations, time period/atmosphere, sensory details, and how the setting relates to the character/plot.
    * **`ThemeChain`:** Identifies abstract themes with brief explanations, linking them to the story's context.
    * **`TitleChain`:** Generates a single novel title based on all preceding information.
    * **`PlotChain`:** Develops a multi-act plot outline. It uses a `HELPER_PROMPT` to first generate compelling story attributes/narrative devices specific to the genre and author style, which are then incorporated into the main plot generation prompt.
    * **`ChaptersChain`:** Creates a list of chapter titles and one-sentence summaries, adhering to a strict output format for reliable parsing. Includes robust parsing logic with regex and fallbacks.
    * **`EventChain`:** For each chapter, generates a numbered list of 3-7 key plot events or scenes that must occur.
    * **`WriterChain`:** This is the core narrative generation unit.
        * **Context-Rich Prompt:** It receives a comprehensive context: character profile, setting, themes, overall plot, current chapter info, history of previously written events (global), and text previously written *within the current chapter*.
        * **Focused Task:** It's instructed to write *only* for the `current_event`.
        * **Style Emulation:** Strongly emphasizes writing in the style of the specified `author`.
        * **"Show, Don't Tell":** A key instruction in its prompt.
    * **`RefinementChain` (Optional):** Takes the drafted text of an entire chapter and attempts to improve it based on a chosen focus (stylistic consistency, prose flow, sensory details, etc.). This is an experimental feature.

4.  **Orchestration Functions:**
    * **`generate_events_for_all_chapters(...)`:** Iterates through the chapter list and calls `EventChain` for each.
    * **`write_book(...)`:** The main loop that iterates through each chapter and then through each event in that chapter, calling `WriterChain` for each event. It accumulates the text for each chapter and optionally calls `RefinementChain`. It manages a `previous_events_history` list and `chapter_paragraphs_accumulator` to provide continuity to the `WriterChain`.

5.  **`DocWriter` Class:**
    * **`_sanitize_filename(self, name)`:** Creates a filesystem-safe filename.
    * **`write_doc(self, ...)`:** Uses the `python-docx` library to create a Word document. It assembles enhanced front matter (title, genre, author, themes, setting description, and a table of contents based on the chapter list) before adding the main novel content, chapter by chapter. Includes page breaks between chapters.

6.  **`main()` Function:**
    * Sets up initial parameters (resume file, subject, author style, genre).
    * Initializes all chain and writer objects.
    * Calls the chains sequentially, handling outputs and passing them as inputs to the next relevant chain.
    * Includes error checking after each major step and can skip subsequent steps if critical information is missing (e.g., plot).
    * Times each major generation step.

### Technologies Used:

* **Python 3.x**
* **Ollama:** For running LLMs locally (e.g., Gemma, Llama 3, Mistral).
* **Langchain (`langchain`, `langchain-ollama`, `langchain-community`):**
    * `OllamaLLM`: Interface to Ollama models.
    * `LLMChain`: To combine prompts and LLMs.
    * `PromptTemplate`: For creating flexible and reusable prompts.
    * `PyPDFLoader`: To load and extract text from PDF files.
* **`python-dotenv`:** To manage environment variables (like Ollama model and URL).
* **`python-docx`:** To create and write `.docx` files.
* **`re` (Regular Expressions):** Extensively used for parsing LLM outputs.
* **Standard Libraries:** `os`, `time`, `traceback`.

## Setup & Prerequisites

1.  **Install Python:** Ensure you have Python 3.8 or newer installed.
2.  **Install Ollama:**
    * Download and install Ollama from [ollama.com](https://ollama.com/).
    * Verify Ollama is running: `ollama list` in your terminal.
3.  **Pull an LLM Model:**
    * You need to have a model pulled that Ollama can serve. Examples:
        ```bash
        ollama pull gemma:2b       # A smaller, faster model
        ollama pull llama3:8b      # A larger, more capable model
        ollama pull mistral        # Another popular model
        ollama pull gemma2:latest  # Or any other preferred model
        ```
    * The script defaults to `gemma2:latest` (configurable via `DEFAULT_MODEL` or `.env`). Make sure your chosen model is pulled.
4.  **Create a Project Directory & Virtual Environment (Recommended):**
    ```bash
    mkdir StoryProject
    cd StoryProject
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
5.  **Clone the Repository (or download `app_v2.py`):**
    ```bash
    git clone [https://github.com/Eeman1113/Story.git](https://github.com/Eeman1113/Story.git)
    cd Story
    ```
    Or, simply place `app_v2.py` in your project directory.

6.  **Create `requirements.txt`:**
    Create a file named `requirements.txt` in the project root with the following content:
    ```txt
    python-dotenv
    python-docx
    langchain
    langchain-ollama
    langchain-community
    pypdf
    ```
7.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
8.  **Prepare Input Files:**
    * Create an output folder (default is `./docs`, will be created if it doesn't exist).
    * Place your resume PDF (e.g., `divi_1.pdf`) inside this `docs` folder. You might need to adjust `DEFAULT_RESUME_FILENAME` in the script if your filename is different.

## Configuration

The script can be configured in two main ways:

1.  **Environment Variables (`.env` file):**
    * Create a `.env` file in the root of your project directory:
        ```env
        OLLAMA_MODEL="gemma2:latest"  # Or your preferred Ollama model, e.g., "llama3:8b"
        OLLAMA_BASE_URL="http://localhost:11434"
        ```
    * The script will load these variables using `python-dotenv`.

2.  **In-Script Constants:**
    * **`DEFAULT_MODEL`:** Fallback model if `OLLAMA_MODEL` is not in `.env`.
    * **`OLLAMA_BASE_URL`:** Fallback Ollama URL.
    * **`OUTPUT_FOLDER`:** Directory where the generated `.docx` file and input resume are expected/placed. Default: `'./docs'`.
    * **`LLM_CALL_DELAY_SECONDS`:** Delay between LLM calls. Default: `0.5`. Increase if hitting rate limits or instability.
    * **`DEFAULT_RESUME_FILENAME`:** The name of the resume PDF file expected in `OUTPUT_FOLDER`. Default: `'divi_1.pdf'`.
    * **`ENABLE_REFINEMENT_PASS`:** Set to `True` to enable the experimental `RefinementChain` pass after each chapter is written. Default: `False`. This will significantly increase LLM usage and generation time.

3.  **Main Execution Parameters (within `main()` function):**
    * **`resume_filename`:** Set programmatically, defaults to `DEFAULT_RESUME_FILENAME`.
    * **`subject`:** The core premise or idea of your story. **Modify this directly in the `main()` function.**
    * **`author_style`:** The author whose style the LLM should try to emulate (e.g., `'Ocean Vuong'`, `'Ernest Hemingway'`). **Modify this directly in the `main()` function.**
    * **`genre`:** The genre(s) of the novel (e.g., `'Psychological Literary Fiction / Nocturnal Realism'`). **Modify this directly in the `main()` function.**

## Usage

1.  **Ensure Ollama is running** with the desired model served.
    ```bash
    ollama serve
    # (In another terminal, or ensure the service is running)
    # ollama list (to check available models)
    ```
2.  **Configure** the script as described above (especially the `subject`, `author_style`, `genre` in `main()`, and ensure your resume PDF is correctly placed and named).
3.  **Run the script:**
    ```bash
    python app_v2.py
    ```
4.  The script will print its progress to the console, including:
    * Initialization details.
    * Outputs from each generation step (profile, setting, themes, title, plot, chapter list).
    * Status updates during event generation and narrative writing.
    * Timings for each major step and total execution time.
5.  Upon completion (which can take a significant amount of time depending on the LLM, hardware, and story length), a `.docx` file will be saved in the `OUTPUT_FOLDER`. The filename will be derived from the generated title, author style, and genre.

## Error Handling & Debugging

* The script includes `try-except` blocks around major operations and LLM calls.
* If an LLM call or parsing fails for a specific component, an error message is usually printed, and often a placeholder or error string is used for that component to allow the rest of the process to continue if possible.
    * For example, if theme generation fails, `"N/A"` might be passed to subsequent chains.
    * If plot generation fails, the script will likely stop before attempting to write chapters.
* Detailed errors, including tracebacks, are printed to the console to aid in debugging.
* `verbose=True` is set for `LLMChain` instances, which means Langchain will print detailed information about the prompts being sent to the LLM and the LLM's raw responses. This is very helpful for debugging prompt issues. Set to `False` in production if the output is too noisy.
* Check the console output carefully if the generated document is incomplete or contains error messages.
* Common issues:
    * Ollama service not running or model not pulled/available.
    * Incorrect `OLLAMA_BASE_URL` or `OLLAMA_MODEL`.
    * Resume PDF not found at the specified path.
    * LLM rate limits or timeouts (increase `request_timeout` in `create_llm` or `LLM_CALL_DELAY_SECONDS`).
    * LLM producing output that the parsing logic cannot handle (may require prompt adjustments or parser improvements).

## Future Enhancements / To-Do

* Allow for more interactive input of story parameters.
* Implement more sophisticated context window management for the `WriterChain` (e.g., summarization of older history).
* Allow for regeneration of specific parts (e.g., a single chapter or event).
* Explore different LLM prompting strategies for even better style emulation and coherence.
* Add options for different output formats (e.g., Markdown, plain text).
* Improve the robustness of parsing LLM outputs, perhaps by asking the LLM to generate JSON.
* Integrate with vector stores for more complex knowledge retrieval or character memory.

---
