import os
import gradio as gr
import asyncio
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from database_module.db import SessionLocal
from database_module.models import ModelEntry
from langchain.chat_models import init_chat_model
# Modify imports section to include all required tools
from database_module import (
    init_db, 
    # get_all_models_handler, 
    # search_models_handler,
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


# Ensure server.py imports and registers these tools:
#   app.register_tool("get_all_models", get_all_models_handler)
#   app.register_tool("search_models", search_models_handler)

# Replace the existing MCP client class with this updated version
class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    async def connect_to_server(self, server_script_path: str = "server.py"):
        """Connect to MCP server"""
        try:
            server_params = StdioServerParameters(
                command="python",
                args=[server_script_path],
                env=None
            )
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )
            await self.session.initialize()
            
            # Get available tools from server
            tools_response = await self.session.list_tools()
            available_tools = [t.name for t in tools_response.tools]
            print("Connected to server with tools:", available_tools)
            
            return True
        except Exception as e:
            print(f"Failed to connect to MCP server: {e}")
            return False

    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the MCP server"""
        if not self.session:
            raise RuntimeError("Not connected to MCP server")
        try:
            response = await self.session.call_tool(tool_name, arguments)
            return response.content
        except Exception as e:
            print(f"Error calling tool {tool_name}: {e}")
            raise

    async def close(self):
        """Close the MCP client connection"""
        if self.session:
            await self.exit_stack.aclose()


# Global MCP client instance
mcp_client = MCPClient()


# Helper to run async functions
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
        result = run_async(mcp_client.call_tool("run_initial_diagnostics", {
            "model": model_name,
            "model_capabilities": capabilities
        }))
        return result
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        return None

def check_model_drift(model_name: str):
    """Check drift for existing model"""
    try:
        result = run_async(mcp_client.call_tool("check_drift", {
            "model": model_name
        }))
        return result
    except Exception as e:
        print(f"Error checking drift: {e}")
        return None

# Initialize MCP connection on startup
def initialize_mcp_connection():
    try:
        run_async(mcp_client.connect_to_server())
        print("Successfully connected to MCP server")
        return True
    except Exception as e:
        print(f"Failed to connect to MCP server: {e}")
        return False


# Wrapper functions remain unchanged but now call real DB-backed MCP tools
def get_models_from_db():
    try:
        result = run_async(mcp_client.call_tool("get_all_models", {}))
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"Error getting models: {e}")
        return []


def get_available_model_names():
    return [m["name"] for m in get_models_from_db()]


def search_models_in_db(search_term: str):
    try:
        result = run_async(mcp_client.call_tool("search_models", {"search_term": search_term}))
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"Error searching models: {e}")
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
    """Get model details from database via MCP"""
    try:
        result = run_async(mcp_client.call_tool("get_model_details", {"model_name": model_name}))
        return result
    except Exception as e:
        print(f"Error getting model details: {e}")
        return {"name": model_name, "system_prompt": "You are a helpful AI assistant.", "description": ""}

def enhance_prompt_via_mcp(prompt: str):
    """Enhance prompt using MCP server"""
    try:
        result = run_async(mcp_client.call_tool("enhance_prompt", {"prompt": prompt}))
        return result.get("enhanced_prompt", prompt)
    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        return f"Enhanced: {prompt}\n\nAdditional context: Be more specific, helpful, and provide detailed responses while maintaining a professional tone."

def save_model_to_db(model_name: str, system_prompt: str):
    """Save model to database via MCP"""
    try:
        result = run_async(mcp_client.call_tool("save_model", {
            "model_name": model_name,
            "system_prompt": system_prompt
        }))
        return result.get("message", "Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
        return f"Error saving model: {e}"

def get_drift_history_from_db(model_name: str):
    """Get drift history from database via MCP"""
    try:
        result = run_async(mcp_client.call_tool("get_drift_history", {"model_name": model_name}))
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"Error getting drift history: {e}")
        # Fallback data for demonstration
        return [
            {"date": "2025-06-01", "drift_score": 0.12},
            {"date": "2025-06-05", "drift_score": 0.18},
            {"date": "2025-06-09", "drift_score": 0.15}
        ]

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
    
    # Available LLM choices for new model creation
    llm_choices = [
        "gemini-1.0-pro",
        "gemini-1.5-pro", 
        "groq-llama-3.1-8b-instant",
        "groq-mixtral-8x7b",
        "groq-gpt4"
    ]
    
    return (
        formatted_items,  # model_dropdown choices
        formatted_items[0] if formatted_items else None,  # model_dropdown value
        llm_choices,  # new_llm choices
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
        outputs=[model_dropdown, model_dropdown, new_model_name, selected_model_display, drift_model_display]
    )

if __name__ == "__main__":
    demo.launch()
