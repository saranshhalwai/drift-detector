import os
import gradio as gr
import asyncio
from typing import Optional, List, Dict
from mcp_agent.core.fastagent import FastAgent

from database_module.db import SessionLocal
from database_module.models import ModelEntry
from langchain.chat_models import init_chat_model
# Modify imports section to include all required tools
from database_module import (
    init_db, 
    get_all_models_handler,
    search_models_handler,
    # save_model_handler,
    # get_model_details_handler,
    # calculate_drift_handler,
    # get_drift_history_handler
)
import json
from datetime import datetime
import plotly.graph_objects as go

# --- Initialize database and MCP tool registration ---
# Create tables and register MCP handlers
init_db()

# Fast Agent client initialization - This is the "scapegoat" client whose drift we're detecting
fast = FastAgent("Scapegoat Client")

@fast.agent(
    name="scapegoat",
    instruction="You are a test client whose drift will be detected and measured over time",
    servers=["drift-server"]
)
async def setup_agent():
    # This function defines the scapegoat agent that will be monitored for drift
    pass

# Global scapegoat client instance to be monitored for drift
scapegoat_client = None

# Initialize the scapegoat client that will be tested for drift
async def initialize_scapegoat_client():
    global scapegoat_client
    print("Initializing scapegoat client for drift monitoring...")
    async with fast.run() as agent:
        scapegoat_client = agent
        return agent
        
# Helper to run async functions with FastAgent
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    else:
        # return result if coroutine returns value, else schedule
        task = loop.create_task(coro)
        return loop.run_until_complete(task) if not task.done() else task

def run_initial_diagnostics(model_name: str, capabilities: str):
    """Run initial diagnostics for a new model"""
    try:
        # Use FastAgent's send method with a formatted message to call the tool
        message = f"""Please call the run_initial_diagnostics tool with the following parameters:
        model: {model_name}
        model_capabilities: {capabilities}
        
        This tool will generate and store baseline diagnostics for the model.
        """

        result = run_async(scapegoat_client(message))
        return result
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        return None

def check_model_drift(model_name: str):
    """Check drift for existing model"""
    try:
        # Use FastAgent's send method with a formatted message to call the tool
        message = f"""Please call the check_drift tool with the following parameters:
        model: {model_name}
        
        This tool will re-run diagnostics and compare to baseline for drift scoring.
        """

        result = run_async(scapegoat_client(message))
        return result
    except Exception as e:
        print(f"Error checking drift: {e}")
        return None

# Initialize MCP connection on startup
def initialize_mcp_connection():
    try:
        run_async(initialize_scapegoat_client())
        print("Successfully connected scapegoat client to MCP server")
        return True
    except Exception as e:
        print(f"Failed to connect scapegoat client to MCP server: {e}")
        return False


# Wrapper functions remain unchanged but now call real DB-backed MCP tools
def get_models_from_db():
    """Get all models from database using direct function call"""
    try:
        # Direct function call to database_module instead of using MCP
        result = get_all_models_handler({})

        if result:
            # Format the result to match the expected structure
            return [
                {
                    "name": model["name"],
                    "description": model.get("description", ""),
                    "created": model.get("created", datetime.now().strftime("%Y-%m-%d"))
                }
                for model in result
            ]
        return []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []


def get_available_model_names():
    return [m["name"] for m in get_models_from_db()]


def search_models_in_db(search_term: str):
    """Search models in database using direct function call"""
    try:
        # Direct function call to database_module instead of using MCP
        result = search_models_handler({"search_term": search_term})

        if result:
            # Format the result to match the expected structure
            return [
                {
                    "name": model["name"],
                    "description": model.get("description", ""),
                    "created": model.get("created", datetime.now().strftime("%Y-%m-%d"))
                }
                for model in result
            ]
        # If no results, return empty list
        return []
    except Exception as e:
        print(f"Error searching models: {e}")
        # Fallback to filtering from all models if there's an error
        return [m for m in get_models_from_db() if search_term.lower() in m["name"].lower()]

def format_dropdown_items(models):
    """Format dropdown items to show model name, creation date, and description preview"""
    formatted_items = []
    model_mapping = {}
    
    for model in models:
        desc_preview = model["description"][:40] + ("..." if len(model["description"]) > 40 else "")
        item_label = f"{model['name']} (Created: {model['created']}) - {desc_preview}"
        formatted_items.append(item_label)
        model_mapping[item_label] = model["name"]
    
    return formatted_items, model_mapping

def extract_model_name_from_dropdown(dropdown_value, model_mapping):
    """Extract actual model name from formatted dropdown value"""
    return model_mapping.get(dropdown_value, dropdown_value.split(" (")[0] if dropdown_value else "")

def get_model_details(model_name: str):
    """Get model details from database via direct DB access (fallback)"""
    try:
        with SessionLocal() as session:
            model_entry = session.query(ModelEntry).filter_by(name=model_name).first()
            if model_entry:
                return {
                    "name": model_entry.name,
                    "description": model_entry.description or "",
                    "system_prompt": model_entry.capabilities.split("\nSystem Prompt: ")[1] if "\nSystem Prompt: " in model_entry.capabilities else "",
                    "created": model_entry.created.strftime("%Y-%m-%d %H:%M:%S") if model_entry.created else ""
                }
            return {"name": model_name, "system_prompt": "You are a helpful AI assistant.", "description": ""}
    except Exception as e:
        print(f"Error getting model details: {e}")
        return {"name": model_name, "system_prompt": "You are a helpful AI assistant.", "description": ""}

def enhance_prompt_via_mcp(prompt: str):
    """Enhance prompt locally since enhance_prompt tool is not available in server.py"""
    # Provide a basic prompt enhancement functionality since server doesn't have it
    enhanced_prompts = {
        "helpful": f"{prompt}\n\nPlease be thorough, helpful, and provide detailed responses.",
        "concise": f"{prompt}\n\nPlease provide concise, direct answers.",
        "technical": f"{prompt}\n\nPlease provide technically accurate and comprehensive responses.",
    }

    if "helpful" in prompt.lower():
        return enhanced_prompts["helpful"]
    elif "concise" in prompt.lower() or "brief" in prompt.lower():
        return enhanced_prompts["concise"]
    elif "technical" in prompt.lower() or "detailed" in prompt.lower():
        return enhanced_prompts["technical"]
    else:
        return f"{prompt}\n\nAdditional context: Be specific, helpful, and provide detailed responses while maintaining a professional tone."

def save_model_to_db(model_name: str, system_prompt: str):
    """Save model to database directly since save_model tool is not available in server.py"""
    try:
        # Check if model already exists
        with SessionLocal() as session:
            existing = session.query(ModelEntry).filter_by(name=model_name).first()
            if existing:
                # Update capabilities to include the new system prompt
                capabilities = existing.capabilities
                if "\nSystem Prompt: " in capabilities:
                    # Replace the system prompt part
                    parts = capabilities.split("\nSystem Prompt: ")
                    capabilities = f"{parts[0]}\nSystem Prompt: {system_prompt}"
                else:
                    # Add system prompt if not present
                    capabilities = f"{capabilities}\nSystem Prompt: {system_prompt}"

                existing.capabilities = capabilities
                existing.updated = datetime.now()
                session.commit()
                return {"message": f"Updated existing model: {model_name}"}
            else:
                # Should not happen as models are registered with capabilities before calling this function
                return {"message": f"Model {model_name} not found. Please register it first."}
    except Exception as e:
        print(f"Error saving model: {e}")
        return {"message": f"Error saving model: {e}"}

def get_drift_history_from_db(model_name: str):
    """Get drift history from database directly without any fallbacks"""
    try:
        from database_module.models import DriftEntry

        with SessionLocal() as session:
            # Query the drift_history table for this model
            drift_entries = session.query(DriftEntry).filter(
                DriftEntry.model_name == model_name
            ).order_by(DriftEntry.date.desc()).all()

            # If no entries found, return empty list
            if not drift_entries:
                return []

            # Convert to the expected format
            history = []
            for entry in drift_entries:
                history.append({
                    "date": entry.date.strftime("%Y-%m-%d"),
                    "drift_score": float(entry.drift_score),
                    "model": entry.model_name
                })

            return history
    except Exception as e:
        print(f"Error getting drift history from database: {e}")
        return []  # Return empty list on error, no fallbacks
def create_drift_chart(drift_history):
    """Create drift chart using plotly"""
    if not drift_history:
        return gr.update(value=None)
    
    dates = [entry["date"] for entry in drift_history]
    scores = [entry["drift_score"] for entry in drift_history]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Drift Score',
        line=dict(color='#ff6b6b', width=3),
        marker=dict(size=8, color='#ff6b6b')
    ))
    
    fig.update_layout(
        title='Model Drift Over Time',
        xaxis_title='Date',
        yaxis_title='Drift Score',
        template='plotly_white',
        height=400,
        showlegend=True
    )
    
    return fig

# Global variable to store model mapping
current_model_mapping = {}

# Gradio interface functions
def update_model_dropdown(search_term):
    """Update dropdown choices based on search term"""
    global current_model_mapping
    
    if search_term.strip():
        models = search_models_in_db(search_term.strip())
    else:
        models = get_models_from_db()
    
    formatted_items, model_mapping = format_dropdown_items(models)
    current_model_mapping = model_mapping
    
    return gr.update(choices=formatted_items, value=formatted_items[0] if formatted_items else None)

def on_model_select(dropdown_value):
    """Handle model selection"""
    if not dropdown_value:
        return "", ""
    
    actual_model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
    return actual_model_name, actual_model_name

def cancel_create_new():
    """Cancel create new model"""
    return [
        gr.update(visible=False),  # create_new_section
        None,  # new_model_name (dropdown)
        "",  # new_system_prompt
        gr.update(visible=False),  # enhanced_prompt_display
        gr.update(visible=False),  # prompt_choice
        gr.update(visible=False),  # save_model_button
        gr.update(visible=False)   # save_status
    ]

def enhance_prompt(original_prompt):
    """Enhance prompt and show options"""
    if not original_prompt.strip():
        return [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        ]
    
    enhanced = enhance_prompt_via_mcp(original_prompt.strip())
    return [
        gr.update(value=enhanced, visible=True),
        gr.update(visible=True),
        gr.update(visible=True)
    ]

def register_model_with_capabilities(model_name: str, capabilities: str):
    """Register a new model with its capabilities in the database"""
    try:
        with SessionLocal() as session:
            model_entry = ModelEntry(
                name=model_name,
                capabilities=capabilities,
                created=datetime.now()
            )
            session.add(model_entry)
            session.commit()
            return True
    except Exception as e:
        print(f"Error registering model: {e}")
        return False


def save_new_model(selected_model_name, selected_llm, original_prompt, enhanced_prompt, choice):
    """Save new model to database"""
    if not selected_model_name or not original_prompt.strip() or not selected_llm:
        return [
            "Please provide model name, LLM selection, and system prompt",
            gr.update(visible=True),
            gr.update()
        ]
    
    final_prompt = enhanced_prompt if choice == "Keep Enhanced" else original_prompt
    
    try:
        # Save the model with LLM capabilities
        capabilities = f"{selected_llm}\nSystem Prompt: {final_prompt}"
        register_model_with_capabilities(selected_model_name, capabilities)
        
        status = save_model_to_db(selected_model_name, final_prompt)
        
        # Run initial diagnostics
        diagnostic_result = run_initial_diagnostics(
            selected_model_name,
            capabilities
        )
        
        if diagnostic_result:
            status = f"{status}\n{diagnostic_result[0].text if isinstance(diagnostic_result, list) else diagnostic_result}"
    except Exception as e:
        status = f"Error saving model: {e}"
    
    # Update dropdown choices
    updated_models = get_models_from_db()
    formatted_items, model_mapping = format_dropdown_items(updated_models)
    global current_model_mapping
    current_model_mapping = model_mapping
    
    return [
        status,
        gr.update(visible=True),
        gr.update(choices=formatted_items)
    ]

def chatbot_response(message, history, dropdown_value):
    """Generate chatbot response using selected model"""
    if not message.strip() or not dropdown_value:
        return history, ""
    
    model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
    model_details = get_model_details(model_name)
    system_prompt = model_details.get("system_prompt", "")
    
    try:
        # Initialize LLM based on model details
        # Get model configuration from database
        with SessionLocal() as session:
            model_entry = session.query(ModelEntry).filter_by(name=model_name).first()
            if not model_entry:
                return history + [[message, "Error: Model not found"]], ""
            
            llm_name = model_entry.capabilities.split("\n")[0] if model_entry.capabilities else "groq-llama-3.1-8b-instant"
        
        # Initialize the LLM using langchain
        llm = init_chat_model(
            llm_name,
            model_provider='groq' if llm_name.startswith('groq') else 'google'
        )
        
        # Format the conversation with system prompt
        formatted_prompt = f"System: {system_prompt}\nUser: {message}"
        
        # Get response from LLM
        response = llm.invoke(formatted_prompt)
        response_text = response.content
        
        history.append([message, response_text])
        return history, ""
        
    except Exception as e:
        error_message = f"Error generating response: {str(e)}"
        history.append([message, error_message])
        return history, ""

def calculate_drift(dropdown_value):
    """Calculate drift for selected model"""
    if not dropdown_value:
        return "Please select a model first"
    
    model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
    
    # First try the drift calculation tool
    try:
        result = check_model_drift(model_name)
        if result and isinstance(result, list):
            return "\n".join(msg.text for msg in result)
    except Exception as e:
        print(f"Error calculating drift: {e}")
        return f"Error calculating drift from server side: {e}"
    
    # Fallback to the simpler drift calculation if needed
    # result = calculate_drift_handler({"model_name": model_name})
    return f"Drift Score: {result.get('drift_score', 0.0):.3f}\n{result.get('message', '')}"

def refresh_drift_history(dropdown_value):
    """Refresh drift history for selected model"""
    if not dropdown_value:
        return [], gr.update(value=None)
    
    model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
    history = get_drift_history_from_db(model_name)
    chart = create_drift_chart(history)
    
    return history, chart

def initialize_interface():
    """Initialize interface with MCP connection and default data"""
    # Connect to MCP server
    mcp_connected = initialize_mcp_connection()
    
    # Get initial model data
    models = get_models_from_db()
    formatted_items, model_mapping = format_dropdown_items(models)
    global current_model_mapping
    current_model_mapping = model_mapping
    
    return (
        formatted_items,  # model_dropdown choices
        formatted_items[0] if formatted_items else None,  # model_dropdown value
        "",  # new_model_name - should be empty string, not choices
        formatted_items[0].split(" (")[0] if formatted_items else "",  # selected_model_display
        formatted_items[0].split(" (")[0] if formatted_items else ""   # drift_model_display
    )


# Create Gradio interface
with gr.Blocks(title="AI Model Management & Interaction Platform") as demo:
    gr.Markdown("# AI Model Management & Interaction Platform")
    
    with gr.Row():
        # Left Column - Model Selection
        with gr.Column(scale=1):
            gr.Markdown("### Model Selection")
            
            model_dropdown = gr.Dropdown(
                choices=[], #work here Here show the already created models (fetched from database using mcp functions defined above)
                label="Select Model",
                interactive=True
            )
            
            search_box = gr.Textbox(
                placeholder="Search by model name or description...",
                label="Search Models"
            )
            
            create_new_button = gr.Button("Create New Model", variant="secondary")
            
            # Create New Model Section (Initially Hidden)
            with gr.Group(visible=False) as create_new_section:
                gr.Markdown("#### Create New Model")
                new_model_name = gr.Textbox(
                    label="Model name",
                    placeholder="Model name"
                )
                new_llm = gr.Dropdown( 
                    choices=[
                        "gemini-1.0-pro",
                        "gemini-1.5-pro",
                        "groq-llama-3.1-8b-instant",
                        "groq-mixtral-8x7b",
                        "groq-gpt4"
                    ], #work here to show options to select llms(available to use) like gemini-1.5-pro, etc google models, groq models (atleast 5 in total)
                    label="Select LLM Name",
                    interactive=True
                )
                new_system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter system prompt",
                    lines=3
                )
                
                with gr.Row():
                    enhance_button = gr.Button("Enhance Prompt", variant="primary")
                    cancel_button = gr.Button("Cancel", variant="secondary")
                
                enhanced_prompt_display = gr.Textbox(
                    label="Enhanced Prompt",
                    interactive=False,
                    lines=4,
                    visible=False
                )
                
                prompt_choice = gr.Radio(
                    choices=["Keep Enhanced", "Keep Original"],
                    label="Choose Prompt to Use",
                    visible=False
                )
                
                save_model_button = gr.Button("Save Model", variant="primary", visible=False)
                save_status = gr.Textbox(label="Status", interactive=False, visible=False)
        
        # Right Column - Model Operations
        with gr.Column(scale=2):
            gr.Markdown("### Model Operations")
            
            with gr.Tabs():
                # Chatbot Tab
                with gr.TabItem("Chatbot"):
                    selected_model_display = gr.Textbox(
                        label="Currently Selected Model",
                        interactive=False
                    )
                    
                    chatbot_interface = gr.Chatbot(height=400)
                    
                    with gr.Row():
                        msg_input = gr.Textbox(
                            placeholder="Enter your message...",
                            label="Message",
                            scale=4
                        )
                        send_button = gr.Button("Send", variant="primary", scale=1)
                    
                    clear_chat = gr.Button("Clear Chat", variant="secondary")
                
                # Drift Analysis Tab
                with gr.TabItem("Drift Analysis"):
                    drift_model_display = gr.Textbox(
                        label="Model for Drift Analysis",
                        interactive=False
                    )
                    
                    with gr.Row():
                        calculate_drift_button = gr.Button("Calculate New Drift", variant="primary")
                        refresh_history_button = gr.Button("Refresh History", variant="secondary")
                    
                    drift_result = gr.Textbox(label="Latest Drift Calculation", interactive=False)
                    
                    gr.Markdown("#### Drift History")
                    drift_history_display = gr.JSON(label="Drift History Data")
                    
                    gr.Markdown("#### Drift Chart")
                    drift_chart = gr.Plot(label="Drift Over Time")

    # Event Handlers
    
    # Search functionality - Dynamic update
    search_box.change(
        update_model_dropdown,
        inputs=[search_box],
        outputs=[model_dropdown]
    )
    
    # Model selection updates
    model_dropdown.change(
        on_model_select,
        inputs=[model_dropdown],
        outputs=[selected_model_display, drift_model_display]
    )
    
    # Create new model functionality
    def show_create_new():
        """Show the create new model section"""
        return gr.update(visible=True), gr.update(value="") 
    
    create_new_button.click(
        show_create_new,
        outputs=[create_new_section, new_model_name]
    )
    
    cancel_button.click(cancel_create_new, outputs=[
        create_new_section, new_model_name, new_system_prompt,
        enhanced_prompt_display, prompt_choice, save_model_button, save_status
    ])
    
    # Enhance prompt
    enhance_button.click(
        enhance_prompt,
        inputs=[new_system_prompt],
        outputs=[enhanced_prompt_display, prompt_choice, save_model_button]
    )
    
    # Save model
    save_model_button.click(
    save_new_model,
    inputs=[new_model_name, new_llm, new_system_prompt, enhanced_prompt_display, prompt_choice],
    outputs=[save_status, save_status, model_dropdown]
)
    
    # Chatbot functionality
    send_button.click(
        chatbot_response,
        inputs=[msg_input, chatbot_interface, model_dropdown],
        outputs=[chatbot_interface, msg_input]
    )
    
    msg_input.submit(
        chatbot_response,
        inputs=[msg_input, chatbot_interface, model_dropdown],
        outputs=[chatbot_interface, msg_input]
    )
    
    clear_chat.click(lambda: [], outputs=[chatbot_interface])
    
    # Drift analysis functionality
    calculate_drift_button.click(
        calculate_drift,
        inputs=[model_dropdown],
        outputs=[drift_result]
    )
    
    refresh_history_button.click(
        refresh_drift_history,
        inputs=[model_dropdown],
        outputs=[drift_history_display, drift_chart]
    )
    
    # Initialize interface on load
    demo.load(
        initialize_interface,
        outputs=[
            model_dropdown,  # dropdown choices
            model_dropdown,  # dropdown value
            new_model_name,  # textbox for new model name (empty string)
            selected_model_display,
            drift_model_display
        ]
    )

if __name__ == "__main__":
    demo.launch()
