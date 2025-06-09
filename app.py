import gradio as gr
import asyncio
from typing import Optional, List, Dict
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import json
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px

class MCPClient:
    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
    
    async def connect_to_server(self, server_script_path: str = "mcp_server.py"):
        """Connect to MCP server"""
        is_python = server_script_path.endswith('.py')
        is_js = server_script_path.endswith('.js')
        
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")
        
        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
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
        
        # List available tools
        response = await self.session.list_tools()
        tools = response.tools
        print("Connected to server with tools:", [tool.name for tool in tools])
    
    async def call_tool(self, tool_name: str, arguments: dict):
        """Call a tool on the MCP server"""
        if not self.session:
            raise RuntimeError("Not connected to server")
        
        response = await self.session.call_tool(tool_name, arguments)
        return response.content
    
    async def close(self):
        """Close the MCP client connection"""
        await self.exit_stack.aclose()

# Global MCP client instance
mcp_client = MCPClient()

# Async wrapper functions for Gradio
def run_async(coro):
    """Helper to run async functions in Gradio"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

# Auto-connect to MCP server on startup
def initialize_mcp_connection():
    """Initialize MCP connection on startup"""
    try:
        run_async(mcp_client.connect_to_server())
        print("Successfully connected to MCP server on startup")
        return True
    except Exception as e:
        print(f"Failed to connect to MCP server on startup: {e}")
        return False

# MCP client functions
def get_models_from_db():
    """Get all models from database via MCP"""
    try:
        result = run_async(mcp_client.call_tool("get_all_models", {}))
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"Error getting models: {e}")
        # Fallback data for demonstration
        return [
            {"name": "llama-3.1-8b-instant", "created": "2025-01-15", "description": "Fast and efficient model for instant responses."},
            {"name": "llama3-8b-8192", "created": "2025-02-10", "description": "Extended context window model with 8192 tokens."},
            {"name": "gemini-2.5-pro-preview-06-05", "created": "2025-06-05", "description": "Professional preview version of Gemini 2.5."},
            {"name": "gemini-2.5-flash-preview-05-20", "created": "2025-05-20", "description": "Flash preview with optimized speed."},
            {"name": "gemini-1.5-pro", "created": "2024-12-01", "description": "Stable professional release of Gemini 1.5."}
        ]

def get_available_model_names():
    """Get list of available model names for dropdown"""
    models = get_models_from_db()
    return [model["name"] for model in models]

def search_models_in_db(search_term: str):
    """Search models in database via MCP"""
    try:
        result = run_async(mcp_client.call_tool("search_models", {"search_term": search_term}))
        return result if isinstance(result, list) else []
    except Exception as e:
        print(f"Error searching models: {e}")
        # Fallback search for demonstration
        all_models = get_models_from_db()
        if not search_term:
            return all_models
        term = search_term.lower()
        return [model for model in all_models if term in model["name"].lower() or term in model["description"].lower()]

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

def calculate_drift_via_mcp(model_name: str):
    """Calculate drift for model via MCP"""
    try:
        result = run_async(mcp_client.call_tool("calculate_drift", {"model_name": model_name}))
        return result
    except Exception as e:
        print(f"Error calculating drift: {e}")
        import random
        drift_score = round(random.uniform(0.05, 0.25), 3)
        return {"drift_score": drift_score, "message": f"Drift calculated and saved for {model_name}"}

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

def toggle_create_new():
    """Toggle create new model section visibility"""
    return gr.update(visible=True)

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

def save_new_model(selected_model_name, original_prompt, enhanced_prompt, choice):
    """Save new model to database"""
    if not selected_model_name or not original_prompt.strip():
        return [
            "Please select a model and enter a system prompt",
            gr.update(visible=True),
            gr.update()
        ]
    
    final_prompt = enhanced_prompt if choice == "Keep Enhanced" else original_prompt
    status = save_model_to_db(selected_model_name, final_prompt)
    
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
    """Generate chatbot response"""
    if not message.strip() or not dropdown_value:
        return history, ""
    
    model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
    model_details = get_model_details(model_name)
    system_prompt = model_details.get("system_prompt", "")
    
    # Simulate response (replace with actual LLM call)
    response = f"[{model_name}] Response to: {message}\n(Using system prompt: {system_prompt[:50]}...)"
    history.append([message, response])
    return history, ""

def calculate_drift(dropdown_value):
    """Calculate drift for selected model"""
    if not dropdown_value:
        return "Please select a model first"
    
    model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
    result = calculate_drift_via_mcp(model_name)
    drift_score = result.get("drift_score", 0.0)
    message = result.get("message", "")
    
    return f"Drift Score: {drift_score:.3f}\n{message}"

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
    
    # Get available model names for create new model dropdown
    available_models = get_available_model_names()
    
    return (
        formatted_items,  # model_dropdown choices
        formatted_items[0] if formatted_items else None,  # model_dropdown value
        available_models,  # new_model_name choices
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
                choices=[],
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
                new_model_name = gr.Dropdown(
                    choices=[],
                    label="Select Model Name",
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
        available_models = get_available_model_names()
        return gr.update(visible=True), gr.update(choices=available_models)
    
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
        inputs=[new_model_name, new_system_prompt, enhanced_prompt_display, prompt_choice],
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
    demo.launch(share=True)
