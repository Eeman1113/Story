import streamlit as st
import os
import tempfile
import time
import traceback
import shutil # Import the shutil library for file copying

# --- Import components from your backend script ---
# (Import logic remains the same)
try:
    from ULTIMATE_POWER import (
        MainCharacterChain, SettingChain, ThemeChain, TitleChain, PlotChain,
        ChaptersChain, EventChain, WriterChain, RefinementChain, DocWriter,
        format_themes_string, sort_chapters, generate_events_for_all_chapters, write_book,
        create_llm,
        DEFAULT_MODEL, OLLAMA_BASE_URL, OUTPUT_FOLDER, DEFAULT_RESUME_FILENAME, LLM_CALL_DELAY_SECONDS
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    st.error(f"Fatal Error: Could not import required components from ULTIMATE_POWER.py. Make sure the file exists in the same directory.")
    st.error(f"Details: {e}")
    IMPORT_SUCCESS = False
except Exception as e:
    st.error(f"An unexpected error occurred during import from ULTIMATE_POWER.py.")
    st.exception(e)
    IMPORT_SUCCESS = False

# --- Streamlit App Configuration ---
# (Remains the same)
st.set_page_config(layout="wide", page_title="Ultimate Power Novel Generator")
st.title("🖋️ Ultimate Power: AI Novel Generation Assistant")
st.markdown("Upload a PDF resume to inspire a character, define your story, and let the AI craft a novel outline and draft!")

# --- Initialize Session State ---
# (Remains the same)
if 'generation_complete' not in st.session_state:
    st.session_state.generation_complete = False
if 'error_message' not in st.session_state:
    st.session_state.error_message = None
if 'generated_data' not in st.session_state:
    st.session_state.generated_data = {}
if 'docx_path' not in st.session_state:
    st.session_state.docx_path = None
if 'temp_pdf_path' not in st.session_state:
    st.session_state.temp_pdf_path = None
if 'backend_copy_pdf_path' not in st.session_state: # Add state for the copied file path
    st.session_state.backend_copy_pdf_path = None

# --- Helper Function for Running Steps ---
# (Remains the same)
def run_generation_step(step_function, step_name, *args, **kwargs):
    """Runs a generation step, displays spinner, handles errors, and stores results."""
    result = None
    error = None
    start_time = time.time()
    try:
        with st.spinner(f"🧠 Running Step: {step_name}..."):
            result = step_function(*args, **kwargs)
            st.session_state.error_message = None # Clear previous errors on success
            duration = time.time() - start_time
            st.info(f"✅ {step_name} completed in {duration:.2f}s.")
    except FileNotFoundError as fnf_error:
        # Be more specific about potential file issues
        error_detail = str(fnf_error)
        if "_temp_resume_for_chain.pdf" in error_detail:
             error = f"File Error during {step_name}: The backend could not find the resume file in the output folder '{cfg_output_folder}'. Check permissions or if the copy succeeded."
        else:
             error = f"File Error during {step_name}: {fnf_error}. Ensure required files exist and are accessible."
        st.error(error)
        traceback.print_exc() # Log traceback to console/logs
    except Exception as e:
        error = f"An error occurred during {step_name}: {e}"
        st.error(error)
        st.exception(e) # Show detailed exception in Streamlit
        traceback.print_exc() # Also log to console

    if error:
        st.session_state.error_message = error
        return None # Indicate failure
    else:
        # (Validation remains the same)
        if isinstance(result, str) and result.strip().startswith("Error:"):
            st.warning(f"Step '{step_name}' reported an error: {result}")
            # Decide if this should halt generation (e.g., critical steps like profile/plot)
            # For profile, let's consider it critical
            if step_name == "Generating Main Character Profile": return None
        elif isinstance(result, dict) and "Error" in result:
             st.warning(f"Step '{step_name}' reported an error: {result['Error']}")
             # Consider it critical based on the step?
        elif not result and step_name in ["Generating Main Character Profile", "Generating Detailed Plot Outline", "Generating Chapter List & Summaries"]:
             st.warning(f"Step '{step_name}' produced empty output. This is critical and will halt generation.")
             return None # Halt on empty critical output

    return result


# --- Sidebar for Configuration ---
# (Remains the same)
st.sidebar.header("⚙️ Configuration")
cfg_ollama_base_url = st.sidebar.text_input("Ollama Base URL", value=OLLAMA_BASE_URL)
cfg_ollama_model = st.sidebar.text_input("Ollama Model", value=DEFAULT_MODEL)
cfg_output_folder = st.sidebar.text_input("Output Folder", value=OUTPUT_FOLDER)
cfg_enable_refinement = st.sidebar.checkbox("Enable Refinement Pass (Experimental, Slower)", value=False)

st.sidebar.markdown("---")
st.sidebar.info(f"Using Model: `{cfg_ollama_model}`\n\nTarget URL: `{cfg_ollama_base_url}`")


# --- Main Area for Inputs ---
# (Remains the same regarding inputs: file upload, text areas, etc.)
st.header("1. Character Inspiration")
uploaded_file = st.file_uploader("Upload Resume (PDF)", type="pdf")

if uploaded_file:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        st.session_state.temp_pdf_path = tmp_file.name
        st.success(f"✅ Resume '{uploaded_file.name}' uploaded and saved temporarily.")
else:
    st.info("Please upload a PDF resume to generate the character profile.")
    # Reset temp path if file is removed
    if st.session_state.temp_pdf_path and os.path.exists(st.session_state.temp_pdf_path):
        try:
            os.remove(st.session_state.temp_pdf_path)
        except Exception: pass
    st.session_state.temp_pdf_path = None
    # Also clear the backend copy path if the source is removed
    if st.session_state.backend_copy_pdf_path and os.path.exists(st.session_state.backend_copy_pdf_path):
         try:
             os.remove(st.session_state.backend_copy_pdf_path)
         except Exception: pass
    st.session_state.backend_copy_pdf_path = None


st.header("2. Story Definition")
subject = st.text_area("Novel Subject / Core Concept", height=150, placeholder="Describe the main idea, conflict, or premise of your story...")
author_style = st.text_input("Author Style Inspiration", placeholder="e.g., Ocean Vuong, N.K. Jemisin, Ted Chiang, Gillian Flynn")
genre = st.text_input("Genre(s)", placeholder="e.g., Psychological Literary Fiction, Sci-Fi Thriller, Epic Fantasy")

st.markdown("---")

# --- Generation Trigger ---
col1, col2 = st.columns([1, 3])
with col1:
    start_button_pressed = st.button("🚀 Generate Novel Components", type="primary", disabled=not uploaded_file or not subject or not author_style or not genre or not IMPORT_SUCCESS)

if start_button_pressed and IMPORT_SUCCESS:
    st.session_state.generation_complete = False
    st.session_state.error_message = None
    st.session_state.generated_data = {}
    st.session_state.docx_path = None
    st.session_state.backend_copy_pdf_path = None # Reset copy path at start

    if not st.session_state.temp_pdf_path or not os.path.exists(st.session_state.temp_pdf_path):
         st.error("Error: Uploaded PDF file not found or not saved correctly.")
         st.stop()

    # --- Set Environment Variables for Ollama ---
    os.environ["OLLAMA_BASE_URL"] = cfg_ollama_base_url
    os.environ["OLLAMA_MODEL"] = cfg_ollama_model

    st.markdown("---")
    st.header("⏳ Generation Process")

    generation_successful = True
    doc_writer = None # Initialize outside loop

    # --- Define path for the copy inside OUTPUT_FOLDER ---
    # Using a fixed, predictable name for the file the backend will look for.
    backend_resume_filename = "_temp_resume_for_chain.pdf"
    backend_resume_path = os.path.join(cfg_output_folder, backend_resume_filename)
    st.session_state.backend_copy_pdf_path = backend_resume_path # Store for cleanup

    try:
        # --- Instantiate Chains ---
        # (Instantiation remains the same)
        main_character_chain = MainCharacterChain()
        setting_chain = SettingChain()
        theme_chain = ThemeChain()
        title_chain = TitleChain()
        plot_chain = PlotChain()
        chapters_chain = ChaptersChain()
        doc_writer = DocWriter(output_folder=cfg_output_folder) # Use configured output folder

        # --- 1. Prepare File for Backend ---
        # Ensure output folder exists and copy the temp file there
        try:
            with st.spinner(f"Preparing resume file for backend (in '{cfg_output_folder}')..."):
                os.makedirs(cfg_output_folder, exist_ok=True) # Ensure target dir exists
                shutil.copy2(st.session_state.temp_pdf_path, backend_resume_path) # Use copy2 to preserve metadata
                st.info(f"Resume copied for backend processing as '{backend_resume_filename}'")
        except Exception as copy_error:
            st.error(f"Fatal Error: Failed to copy resume to output folder '{cfg_output_folder}': {copy_error}")
            st.exception(copy_error)
            # Store error and stop, cleanup will happen in finally block
            st.session_state.error_message = f"Failed to copy resume: {copy_error}"
            generation_successful = False
            st.stop() # Stop execution here if copy fails

        # --- 2. Generate Profile ---
        profile = None
        if generation_successful: # Check flag again (though st.stop() should prevent reaching here if copy failed)
            # ***** MODIFIED CALL *****
            # Call run with file_name (basename) instead of text
            profile = run_generation_step(
                main_character_chain.run,
                "Generating Main Character Profile",
                # Pass arguments by keyword, matching backend signature
                file_name=backend_resume_filename, # Pass the filename within OUTPUT_FOLDER
                genre=genre
            )
            if profile:
                st.session_state.generated_data['profile'] = profile
            else:
                # run_generation_step handles error display, just update flag
                generation_successful = False # Profile is critical


        # --- 3. Generate Setting ---
        # (No change needed here, depends on profile variable)
        setting = None
        if generation_successful and profile:
            setting = run_generation_step(
                setting_chain.run,
                "Generating Setting Description",
                subject=subject, genre=genre, profile=profile
            )
            if setting:
                st.session_state.generated_data['setting'] = setting
            else:
                st.warning("Setting generation failed or produced empty result. Continuing...")
                st.session_state.generated_data['setting'] = "Error: Setting generation failed."


        # --- 4. Generate Themes ---
        # (No change needed here)
        themes_dict = None
        themes_str = "N/A"
        if generation_successful and profile and setting:
            themes_dict = run_generation_step(
                theme_chain.run,
                "Generating Core Themes",
                subject=subject, genre=genre, profile=profile, setting=setting or "N/A"
            )
            if themes_dict and "Error" not in themes_dict:
                st.session_state.generated_data['themes_dict'] = themes_dict
                themes_str = format_themes_string(themes_dict)
                st.session_state.generated_data['themes_str'] = themes_str
            else:
                 st.warning("Theme generation failed or produced empty result. Continuing...")
                 st.session_state.generated_data['themes_dict'] = {"Error": "Theme generation failed."}
                 st.session_state.generated_data['themes_str'] = "N/A"


        # --- 5. Generate Title ---
        # (No change needed here)
        title = None
        if generation_successful:
            title = run_generation_step(
                title_chain.run,
                "Generating Novel Title",
                subject=subject, genre=genre, author=author_style,
                profile=profile or "N/A", setting=setting or "N/A", themes_str=themes_str
            )
            if title and not title.startswith("Error") and "Placeholder" not in title:
                st.session_state.generated_data['title'] = title
            else:
                 st.warning(f"Title generation failed or resulted in a placeholder: '{title}'. Using placeholder.")
                 st.session_state.generated_data['title'] = title or "Error Generating Title"


        # --- 6. Generate Plot ---
        # (No change needed here)
        plot = None
        if generation_successful and profile: # Need at least profile
            plot = run_generation_step(
                plot_chain.run,
                "Generating Detailed Plot Outline",
                subject=subject, genre=genre, author=author_style,
                profile=profile, title=st.session_state.generated_data.get('title', "Untitled"),
                setting=setting or "N/A", themes_str=themes_str
            )
            if plot:
                st.session_state.generated_data['plot'] = plot
            else:
                generation_successful = False # Plot is critical


        # --- 7. Generate Chapters ---
        # (No change needed here)
        chapter_dict_sorted = None
        if generation_successful and plot:
            raw_chapter_dict = run_generation_step(
                chapters_chain.run,
                "Generating Chapter List & Summaries",
                subject=subject, genre=genre, author=author_style, profile=profile,
                title=st.session_state.generated_data.get('title', "Untitled"), plot=plot,
                setting=setting or "N/A", themes_str=themes_str
            )
            if raw_chapter_dict:
                chapter_dict_sorted = raw_chapter_dict
                st.session_state.generated_data['chapter_dict_sorted'] = chapter_dict_sorted
            else:
                generation_successful = False # Chapters are critical


        # --- 8. Generate Events ---
        # (No change needed here)
        event_dict = None
        if generation_successful and chapter_dict_sorted:
            with st.spinner("Generating events for all chapters..."):
                 event_dict = generate_events_for_all_chapters(
                     plot, profile, themes_str, chapter_dict_sorted, author_style
                 )
            if event_dict and any(event_dict.values()):
                 st.session_state.generated_data['event_dict'] = event_dict
                 st.info("✅ Events generated for chapters.")
            else:
                 st.warning("Failed to generate events for some or all chapters. Book content might be incomplete.")
                 st.session_state.generated_data['event_dict'] = event_dict or {}


        # --- 9. Write Book Content ---
        # (No change needed here)
        book_content = None
        if generation_successful and chapter_dict_sorted and event_dict is not None:
             with st.spinner(f"✍️ Writing full book content ({len(chapter_dict_sorted)} chapters)... This may take a while!"):
                 book_content = write_book(
                     genre, author_style, st.session_state.generated_data.get('title', "Untitled"),
                     profile, plot, setting or "N/A", themes_str,
                     chapter_dict_sorted, event_dict,
                     refine_chapters=cfg_enable_refinement
                 )
             if book_content:
                 st.session_state.generated_data['book_content'] = book_content
                 st.info("✅ Book content generated.")
             else:
                 st.error("❌ Book content generation failed.")
                 generation_successful = False


        # --- 10. Save to Document ---
        # (No change needed here)
        if generation_successful and book_content and doc_writer:
            saved_path = run_generation_step(
                doc_writer.write_doc,
                "Saving Document to DOCX",
                book_content=book_content,
                chapter_dict=chapter_dict_sorted,
                title=st.session_state.generated_data.get('title', "Untitled"),
                genre=genre, author=author_style,
                themes_dict=st.session_state.generated_data.get('themes_dict', {"Error":"N/A"}),
                setting=setting or "N/A"
            )
            if saved_path and os.path.exists(saved_path):
                st.session_state.docx_path = saved_path
                st.success(f"🎉 Novel document generated successfully!")
            else:
                st.error("Failed to save the document.")
                generation_successful = False


        # --- Final State ---
        st.session_state.generation_complete = generation_successful

    except Exception as final_error:
        st.error(f"A critical error occurred during the generation pipeline:")
        st.exception(final_error)
        st.session_state.error_message = str(final_error)
        st.session_state.generation_complete = False

    finally:
        # --- Combined Cleanup ---
        # Ensure both temp file and backend copy are attempted to be removed

        # Cleanup Copied Resume File (if it exists)
        backend_copy_to_clean = st.session_state.get('backend_copy_pdf_path')
        if backend_copy_to_clean and os.path.exists(backend_copy_to_clean):
             try:
                 time.sleep(0.2) # Small delay might help on some systems
                 os.remove(backend_copy_to_clean)
                 print(f"Debug: Cleaned up backend copy: {backend_copy_to_clean}") # Use print for debug
                 st.session_state.backend_copy_pdf_path = None # Clear state after removal
             except PermissionError:
                 st.warning(f"Could not delete backend resume copy '{backend_copy_to_clean}' due to permissions. Manual cleanup might be needed in '{cfg_output_folder}'.")
             except Exception as cleanup_copy_error:
                 st.warning(f"Error during backend copy cleanup: {cleanup_copy_error}")
                 # Keep state? Or clear anyway? Clear it.
                 st.session_state.backend_copy_pdf_path = None

        # Cleanup Original Temporary PDF (if it exists)
        temp_to_clean = st.session_state.get('temp_pdf_path')
        if temp_to_clean and os.path.exists(temp_to_clean):
            try:
                time.sleep(0.2)
                os.remove(temp_to_clean)
                print(f"Debug: Cleaned up temp file: {temp_to_clean}") # Use print for debug
                st.session_state.temp_pdf_path = None # Clear state
            except PermissionError:
                st.warning(f"Could not delete temporary file {temp_to_clean} due to permissions. Manual cleanup might be needed.")
            except Exception as cleanup_error:
                st.warning(f"Error during temporary file cleanup: {cleanup_error}")
                st.session_state.temp_pdf_path = None # Clear state


# --- Display Results ---
# (Remains the same)
st.markdown("---")
st.header("💡 Generated Novel Components")

if not st.session_state.generated_data and not st.session_state.error_message:
     st.info("Click 'Generate Novel Components' after providing inputs.")
elif st.session_state.error_message and not st.session_state.generation_complete:
     st.error(f"Generation failed. Last error: {st.session_state.error_message}")

if st.session_state.generated_data:
    data = st.session_state.generated_data

    # (Display logic for profile, setting, themes, title, plot, chapters remains the same)
    if 'profile' in data:
        with st.expander("👤 Main Character Profile", expanded=False):
            st.markdown(data['profile'])
    # ... etc ...
    if 'chapter_dict_sorted' in data:
        with st.expander("📖 Chapter List", expanded=False):
             if isinstance(data['chapter_dict_sorted'], dict):
                 for title, desc in data['chapter_dict_sorted'].items():
                      st.markdown(f"**{title}:** {desc}")
             else:
                 st.warning("Chapter data is not in the expected format.")


    # --- Download Button ---
    # (Remains the same)
    if st.session_state.docx_path and os.path.exists(st.session_state.docx_path):
        st.markdown("---")
        st.subheader("✅ Download Your Novel")
        try:
            with open(st.session_state.docx_path, "rb") as fp:
                st.download_button(
                    label="Download Novel (.docx)",
                    data=fp,
                    file_name=os.path.basename(st.session_state.docx_path),
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
        except Exception as download_err:
            st.error(f"Error preparing download: {download_err}")
    elif st.session_state.generation_complete:
         st.warning("Document saving might have failed or content generation was incomplete. Download not available.")

elif not start_button_pressed and not st.session_state.error_message:
    st.markdown("*Results will appear here after generation.*")

# --- Footer ---
# (Remains the same)
st.markdown("---")
st.caption("Ultimate Power Novel Generator - Using Streamlit and Langchain with Ollama.")