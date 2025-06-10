import os
import gradio as gr
import asyncio
from typing import Optional, List, Dict
import subprocess
import time
import signal
import sys
# Add these imports at the top of your Gradio file
from database_module.mcp_tools import (
    get_drift_history_handler,
    calculate_drift_handler
)
import threading
import concurrent.futures
# Add these imports at the top of your Gradio file
from ourllm import llm  # Import the actual LLM instance
from dotenv import load_dotenv
# Add error handling for imports
try:
    from database_module.db import SessionLocal
    from database_module.models import ModelEntry
    from langchain.chat_models import init_chat_model
    from database_module import (
        init_db,
        get_all_models_handler,
        search_models_handler,
    )

    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Database modules not available: {e}")
    print("⚠️ Running in demo mode without database functionality")
    DATABASE_AVAILABLE = False

import json
from datetime import datetime
import plotly.graph_objects as go
try:
    from ourllm import llm
    print("✅ Successfully imported LLM from ourllm.py")
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"❌ Failed to import LLM: {e}")
    LLM_AVAILABLE = False

# Mock database functions for when database is not available
def mock_init_db():
    print("📝 Mock database initialized")
    return True


def mock_get_all_models():
    return [
        {"name": "demo-model-1", "description": "Demo model for testing", "created": "2024-01-01"},
        {"name": "demo-model-2", "description": "Another demo model", "created": "2024-01-02"}
    ]


def mock_search_models(search_term):
    all_models = mock_get_all_models()
    return [m for m in all_models if search_term.lower() in m["name"].lower()]


def mock_register_model(model_name, capabilities):
    print(f"📝 Mock: Registered model {model_name}")
    return True


# Use mock functions if database is not available
if not DATABASE_AVAILABLE:
    init_db = mock_init_db
    get_all_models_handler = lambda x: mock_get_all_models()
    search_models_handler = lambda x: mock_search_models(x.get("search_term", ""))

# Initialize database (or mock)
try:
    init_db()
    print("✅ Database initialization successful")
except Exception as e:
    print(f"⚠️ Database initialization failed: {e}")
    DATABASE_AVAILABLE = False

# Global variables
scapegoat_client = None
server_manager = None
current_model_mapping = {}


# --- Simplified Database Functions ---
def ensure_database_setup():
    """Ensure database is properly set up"""
    if not DATABASE_AVAILABLE:
        print("✅ Running in demo mode - no database required")
        return True

    try:
        # Test database connection
        with SessionLocal() as session:
            session.execute("SELECT 1")
            session.commit()
            print("✅ Database connection successful")
            return True
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False


def register_model_with_capabilities(model_name: str, capabilities: str):
    """Register a new model with its capabilities"""
    if not DATABASE_AVAILABLE:
        return mock_register_model(model_name, capabilities)

    try:
        with SessionLocal() as session:
            existing = session.query(ModelEntry).filter_by(name=model_name).first()
            if existing:
                existing.capabilities = capabilities
                existing.updated = datetime.now()
                session.commit()
                print(f"✅ Updated existing model: {model_name}")
            else:
                model_entry = ModelEntry(
                    name=model_name,
                    capabilities=capabilities,
                    created=datetime.now()
                )
                session.add(model_entry)
                session.commit()
                print(f"✅ Registered new model: {model_name}")
            return True
    except Exception as e:
        print(f"❌ Error registering model: {e}")
        return False


# --- Simplified Model Management Functions ---
def get_models_from_db():
    """Get all models from database"""
    if not DATABASE_AVAILABLE:
        return mock_get_all_models()

    try:
        result = get_all_models_handler({})
        if result:
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
        print(f"❌ Error getting models: {e}")
        return mock_get_all_models()


load_dotenv()


# Replace your current chatbot_response function with this:
def chatbot_response(message, history, dropdown_value):
    """Generate chatbot response using actual LLM with debug info"""
    print(f"🔍 DEBUG: Function called with message: '{message}'")
    print(f"🔍 DEBUG: LLM_AVAILABLE: {LLM_AVAILABLE}")
    print(f"🔍 DEBUG: GROQ_API_KEY exists: {'GROQ_API_KEY' in os.environ}")

    if not message or not message.strip() or not dropdown_value:
        print("🔍 DEBUG: Empty message or dropdown")
        return history, ""

    try:
        model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
        print(f"🔍 DEBUG: Model name: {model_name}")

        # Initialize history if needed
        if history is None:
            history = []

        # Check if LLM is available and API key is set
        if not LLM_AVAILABLE:
            response_text = "❌ LLM not available - check ourllm.py import"
        elif not os.getenv("GROQ_API_KEY"):
            response_text = "❌ GROQ_API_KEY not found in environment variables"
        else:
            try:
                print("🔍 DEBUG: Attempting to call LLM...")

                # Simple direct call to LLM
                response = llm.invoke(message)
                response_text = str(response.content).strip()

                print(f"🔍 DEBUG: LLM response received: {response_text[:100]}...")

                if not response_text:
                    response_text = "❌ LLM returned empty response"

            except Exception as e:
                print(f"🔍 DEBUG: LLM call failed: {e}")
                response_text = f"❌ LLM Error: {str(e)}"

        # Add to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})

        print(f"🔍 DEBUG: Final response: {response_text}")
        return history, ""

    except Exception as e:
        print(f"🔍 DEBUG: General error in chatbot_response: {e}")
        if history is None:
            history = []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""

def search_models_in_db(search_term: str):
    """Search models in database"""
    if not DATABASE_AVAILABLE:
        return mock_search_models(search_term)

    try:
        result = search_models_handler({"search_term": search_term})
        if result:
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
        print(f"❌ Error searching models: {e}")
        return [m for m in get_models_from_db() if search_term.lower() in m["name"].lower()]


def format_dropdown_items(models):
    """Format dropdown items"""
    if not models:
        return [], {}

    formatted_items = []
    model_mapping = {}

    for model in models:
        desc_preview = model["description"][:40] + ("..." if len(model["description"]) > 40 else "")
        item_label = f"{model['name']} (Created: {model['created']}) - {desc_preview}"
        formatted_items.append(item_label)
        model_mapping[item_label] = model["name"]

    return formatted_items, model_mapping


def extract_model_name_from_dropdown(dropdown_value, model_mapping):
    """Extract model name from dropdown"""
    if not dropdown_value:
        return ""
    return model_mapping.get(dropdown_value, dropdown_value.split(" (")[0] if dropdown_value else "")


def get_model_details(model_name: str):
    """Get model details from database"""
    try:
        if DATABASE_AVAILABLE:
            with SessionLocal() as session:
                model_entry = session.query(ModelEntry).filter_by(name=model_name).first()
                if model_entry:
                    return {
                        "name": model_entry.name,
                        "description": model_entry.description or "",
                        "system_prompt": model_entry.capabilities.split("System Prompt: ")[
                            1] if model_entry.capabilities and "System Prompt: " in model_entry.capabilities else "You are a helpful AI assistant.",
                        "created": model_entry.created.strftime("%Y-%m-%d %H:%M:%S") if model_entry.created else ""
                    }
        return {"name": model_name, "system_prompt": "You are a helpful AI assistant.", "description": "Demo model"}
    except Exception as e:
        print(f"❌ Error getting model details: {e}")
        return {"name": model_name, "system_prompt": "You are a helpful AI assistant.", "description": "Demo model"}


# --- Gradio Interface Functions ---
def update_model_dropdown(search_term):
    """Update dropdown based on search"""
    global current_model_mapping

    try:
        if search_term and search_term.strip():
            models = search_models_in_db(search_term.strip())
        else:
            models = get_models_from_db()

        formatted_items, model_mapping = format_dropdown_items(models)
        current_model_mapping = model_mapping

        # Return dropdown with proper value handling
        if formatted_items:
            return gr.update(choices=formatted_items, value=formatted_items[0])
        else:
            return gr.update(choices=[], value=None)
    except Exception as e:
        print(f"❌ Error updating dropdown: {e}")
        return gr.update(choices=[], value=None)


def on_model_select(dropdown_value):
    """Handle model selection"""
    if not dropdown_value or not current_model_mapping:
        return "", ""

    try:
        actual_model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
        return actual_model_name, actual_model_name
    except Exception as e:
        print(f"❌ Error in model selection: {e}")
        return "", ""


def show_create_new():
    """Show create new model section"""
    return gr.update(visible=True), gr.update(value="")


def cancel_create_new():
    """Cancel create new model"""
    return [
        gr.update(visible=False),  # create_new_section
        "",  # new_model_name
        "",  # new_system_prompt
        gr.update(visible=False),  # enhanced_prompt_display
        gr.update(visible=False),  # prompt_choice
        gr.update(visible=False),  # save_model_button
        gr.update(visible=False)  # save_status
    ]


def enhance_prompt(original_prompt):
    """Enhance prompt locally"""
    if not original_prompt or not original_prompt.strip():
        return [
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        ]

    enhanced = f"{original_prompt}\n\nAdditional context: Be specific, helpful, and provide detailed responses while maintaining a professional tone."
    return [
        gr.update(value=enhanced, visible=True),
        gr.update(visible=True),
        gr.update(visible=True)
    ]


def save_new_model(model_name, selected_llm, original_prompt, enhanced_prompt, choice):
    """Save new model"""
    global current_model_mapping

    if not model_name or not original_prompt or not original_prompt.strip() or not selected_llm:
        return [
            "❌ Please provide model name, LLM selection, and system prompt",
            gr.update(visible=True),
            gr.update()
        ]

    try:
        final_prompt = enhanced_prompt if choice == "Keep Enhanced" else original_prompt
        capabilities = f"{selected_llm}\nSystem Prompt: {final_prompt}"

        if register_model_with_capabilities(model_name, capabilities):
            status = f"✅ Model '{model_name}' saved successfully!"

            # Update dropdown with new models
            updated_models = get_models_from_db()
            formatted_items, model_mapping = format_dropdown_items(updated_models)
            current_model_mapping = model_mapping

            dropdown_update = gr.update(choices=formatted_items, value=formatted_items[0] if formatted_items else None)
        else:
            status = "❌ Error saving model to database"
            dropdown_update = gr.update()

    except Exception as e:
        status = f"❌ Error saving model: {e}"
        dropdown_update = gr.update()

    return [
        status,
        gr.update(visible=True),
        dropdown_update
    ]


# Also add this function to help debug database connection:
def test_database_connection():
    """Test if database connection is working and has data"""
    try:
        if not DATABASE_AVAILABLE:
            return "⚠️ Database not available - running in demo mode"

        # Test getting models
        models = get_all_models_handler({})
        model_count = len(models) if models else 0

        # Test getting drift history for first model if available
        drift_info = ""
        if models and len(models) > 0:
            first_model = models[0]["name"]
            drift_history = get_drift_history_handler({"model_name": first_model})
            drift_count = len(drift_history) if drift_history else 0
            drift_info = f"\n📊 Drift records for '{first_model}': {drift_count}"

        return f"✅ Database connected\n📝 Total models: {model_count}{drift_info}"

    except Exception as e:
        return f"❌ Database test failed: {e}"
# Replace the chatbot_response function in your Gradio file with this:

def chatbot_response(message, history, dropdown_value):
    """Generate chatbot response using actual LLM"""
    print(f"🔍 DEBUG: Function called with message: '{message}'")
    print(f"🔍 DEBUG: LLM_AVAILABLE: {LLM_AVAILABLE}")
    print(f"🔍 DEBUG: GROQ_API_KEY exists: {'GROQ_API_KEY' in os.environ}")

    if not message or not message.strip() or not dropdown_value:
        print("🔍 DEBUG: Empty message or dropdown")
        return history, ""

    try:
        model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
        print(f"🔍 DEBUG: Model name: {model_name}")

        # Initialize history if needed
        if history is None:
            history = []

        # Check if LLM is available and API key is set
        if not LLM_AVAILABLE:
            response_text = "❌ LLM not available - check ourllm.py import"
        elif not os.getenv("GROQ_API_KEY"):
            response_text = "❌ GROQ_API_KEY not found in environment variables"
        else:
            try:
                print("🔍 DEBUG: Attempting to call LLM...")

                # Get model details to use system prompt if available
                model_details = get_model_details(model_name)
                system_prompt = model_details.get("system_prompt", "You are a helpful AI assistant.")

                # Create a message with system context
                full_message = f"System: {system_prompt}\n\nUser: {message}"

                # Call the LLM
                response = llm.invoke(full_message)
                response_text = str(response.content).strip()

                print(f"🔍 DEBUG: LLM response received: {response_text[:100]}...")

                if not response_text:
                    response_text = "❌ LLM returned empty response"

            except Exception as e:
                print(f"🔍 DEBUG: LLM call failed: {e}")
                response_text = f"❌ LLM Error: {str(e)}"

        # Add to history
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response_text})

        print(f"🔍 DEBUG: Final response: {response_text}")
        return history, ""

    except Exception as e:
        print(f"🔍 DEBUG: General error in chatbot_response: {e}")
        if history is None:
            history = []
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})
        return history, ""


# Also add this helper function to test LLM connectivity:
def test_llm_connection():
    """Test if LLM is working properly"""
    try:
        if not LLM_AVAILABLE:
            return "❌ LLM not imported"

        if not os.getenv("GROQ_API_KEY"):
            return "❌ GROQ_API_KEY not found"

        # Test with a simple message
        response = llm.invoke("Hello, please respond with 'LLM is working'")
        return f"✅ LLM working: {response.content}"
    except Exception as e:
        return f"❌ LLM test failed: {e}"


# Add this to your interface initialization to test LLM on startup:
# Add this to your interface initialization to show database status
def initialize_interface_with_debug():
    """Initialize interface with database debug info"""
    global current_model_mapping

    # Test database connection
    db_status = test_database_connection()
    print(f"🔍 Database Status: {db_status}")

    try:
        models = get_models_from_db()
        formatted_items, model_mapping = format_dropdown_items(models)
        current_model_mapping = model_mapping

        if formatted_items:
            dropdown_value = formatted_items[0]
            first_model_name = extract_model_name_from_dropdown(dropdown_value, model_mapping)
            dropdown_update = gr.update(choices=formatted_items, value=dropdown_value)
        else:
            dropdown_value = None
            first_model_name = ""
            dropdown_update = gr.update(choices=[], value=None)

        return (
            dropdown_update,
            "",
            first_model_name,
            first_model_name
        )
    except Exception as e:
        print(f"❌ Error initializing interface: {e}")
        return (
            gr.update(choices=[], value=None),
            "",
            "",
            ""
        )

# Replace your existing functions with these corrected versions:

def calculate_drift(dropdown_value):
    """Calculate drift for model - using actual database"""
    if not dropdown_value:
        return "❌ Please select a model first"

    try:
        model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)

        if not DATABASE_AVAILABLE:
            # Fallback for demo mode
            import random
            drift_score = random.randint(10, 80)
            alert = "🚨 Significant drift detected!" if drift_score > 50 else "✅ Drift within acceptable range"
            return f"Drift analysis for {model_name}:\nDrift Score: {drift_score}/100\n{alert}"

        # Use actual database function
        result = calculate_drift_handler({"model_name": model_name})

        if "drift_score" in result:
            drift_score = result["drift_score"]
            # Convert to percentage if it's a decimal
            if isinstance(drift_score, float) and drift_score <= 1.0:
                drift_score = int(drift_score * 100)

            alert = "🚨 Significant drift detected!" if drift_score > 50 else "✅ Drift within acceptable range"
            return f"Drift analysis for {model_name}:\nDrift Score: {drift_score}/100\n{alert}\n\n{result.get('message', '')}"
        else:
            return f"❌ Error calculating drift: {result.get('message', 'Unknown error')}"

    except Exception as e:
        print(f"❌ Error calculating drift: {e}")
        return f"❌ Error calculating drift: {str(e)}"


def create_drift_chart(drift_history):
    """Create drift chart from actual data with improved data handling"""
    try:
        if not drift_history or len(drift_history) == 0:
            # Empty chart if no data
            fig = go.Figure()
            fig.add_annotation(
                text="No drift data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                showarrow=False,
                font=dict(size=16)
            )
            fig.update_layout(
                title='Model Drift Over Time - No Data',
                template='plotly_white',
                height=400,
                xaxis_title='Date',
                yaxis_title='Drift Score (%)'
            )
            return fig

        print(f"🔍 DEBUG: Processing {len(drift_history)} drift records")

        # Extract dates and scores from actual data
        dates = []
        scores = []

        for i, entry in enumerate(drift_history):
            print(f"🔍 DEBUG: Processing entry {i}: {entry}")

            # Handle different date formats
            date_value = entry.get("date", entry.get("created_at", entry.get("timestamp", "")))

            if date_value:
                if isinstance(date_value, str):
                    try:
                        from datetime import datetime
                        # Try different date formats
                        if "T" in date_value:
                            # ISO format with time
                            date_obj = datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                            formatted_date = date_obj.strftime("%Y-%m-%d")
                        elif "-" in date_value and len(date_value) >= 10:
                            # YYYY-MM-DD format
                            date_obj = datetime.strptime(date_value[:10], "%Y-%m-%d")
                            formatted_date = date_obj.strftime("%Y-%m-%d")
                        else:
                            # Use as-is if can't parse
                            formatted_date = str(date_value)[:10]
                    except Exception as date_error:
                        print(f"⚠️ Date parsing error for '{date_value}': {date_error}")
                        formatted_date = f"Entry {i + 1}"
                else:
                    # Handle datetime objects
                    try:
                        formatted_date = date_value.strftime("%Y-%m-%d")
                    except:
                        formatted_date = str(date_value)
            else:
                formatted_date = f"Entry {i + 1}"

            dates.append(formatted_date)

            # Handle drift score - try multiple possible field names
            score = entry.get("drift_score", entry.get("score", entry.get("drift", 0)))

            if isinstance(score, str):
                try:
                    score = float(score)
                except ValueError:
                    print(f"⚠️ Could not convert score '{score}' to float, using 0")
                    score = 0
            elif score is None:
                score = 0

            # Convert decimal to percentage if needed
            if isinstance(score, (int, float)):
                if 0 <= score <= 1:
                    score = score * 100  # Convert decimal to percentage
                score = max(0, min(100, score))  # Clamp between 0-100
            else:
                score = 0

            scores.append(score)
            print(f"🔍 DEBUG: Added point - Date: {formatted_date}, Score: {score}")

        print(f"🔍 DEBUG: Final data - Dates: {dates}, Scores: {scores}")

        if len(dates) == 0 or len(scores) == 0:
            raise ValueError("No valid data points found")

        # Create the plot
        fig = go.Figure()

        # Add the main drift line
        fig.add_trace(go.Scatter(
            x=dates,
            y=scores,
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='#ff6b6b', width=3),
            marker=dict(
                size=10,
                color='#ff6b6b',
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>Date:</b> %{x}<br><b>Drift Score:</b> %{y:.1f}%<extra></extra>',
            connectgaps=True  # Connect points even if there are gaps
        ))

        # Add threshold line at 50%
        fig.add_hline(
            y=50,
            line_dash="dash",
            line_color="orange",
            line_width=2,
            annotation_text="Alert Threshold (50%)",
            annotation_position="bottom right"
        )

        # Add another threshold at 75% for critical level
        fig.add_hline(
            y=75,
            line_dash="dot",
            line_color="red",
            line_width=2,
            annotation_text="Critical Threshold (75%)",
            annotation_position="top right"
        )

        # Update layout with better formatting
        fig.update_layout(
            title=f'Model Drift Over Time ({len(drift_history)} data points)',
            xaxis_title='Date',
            yaxis_title='Drift Score (%)',
            template='plotly_white',
            height=450,
            showlegend=True,
            yaxis=dict(
                range=[0, 100],  # Set Y-axis range from 0 to 100%
                ticksuffix='%'
            ),
            xaxis=dict(
                tickangle=45 if len(dates) > 5 else 0,  # Angle labels for many dates
                type='category'  # Treat dates as categories for better spacing
            ),
            hovermode='x unified',  # Better hover experience
            margin=dict(b=100)  # More bottom margin for angled labels
        )

        # Add grid for better readability
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

        return fig

    except Exception as e:
        print(f"❌ Error creating drift chart: {e}")
        print(f"❌ Drift history data: {drift_history}")

        # Return error chart
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart:\n{str(e)}\n\nCheck console for details",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=14, color="red"),
            align="center"
        )
        fig.update_layout(
            title='Error Creating Drift Chart',
            template='plotly_white',
            height=400,
            xaxis_title='Date',
            yaxis_title='Drift Score (%)'
        )
        return fig


def debug_drift_data(drift_history):
    """Helper function to debug drift history data structure"""
    print("🔍 DEBUG: Drift History Analysis")
    print(f"Type: {type(drift_history)}")
    print(f"Length: {len(drift_history) if drift_history else 0}")

    if drift_history:
        for i, entry in enumerate(drift_history[:3]):  # Show first 3 entries
            print(f"Entry {i}: {entry}")
            print(f"  Keys: {list(entry.keys()) if isinstance(entry, dict) else 'Not a dict'}")

    return drift_history


def refresh_drift_history(dropdown_value):
    """Refresh drift history with improved debugging"""
    if not dropdown_value:
        return [], gr.update(value=None)

    try:
        model_name = extract_model_name_from_dropdown(dropdown_value, current_model_mapping)
        print(f"🔍 DEBUG: Getting drift history for model: {model_name}")

        if not DATABASE_AVAILABLE:
            # Enhanced mock data for demo mode
            from datetime import datetime, timedelta
            base_date = datetime.now() - timedelta(days=10)

            history = []
            for i in range(6):  # Create 6 data points
                date_obj = base_date + timedelta(days=i * 2)
                score = 20 + (i * 15) + (i % 2 * 10)  # Varied scores: 20, 45, 50, 75, 70, 95
                history.append({
                    "date": date_obj.strftime("%Y-%m-%d"),
                    "drift_score": min(95, score),  # Cap at 95
                    "model_name": model_name
                })

            print(f"🔍 DEBUG: Generated {len(history)} mock drift records")
        else:
            # Get actual drift history from database
            history_result = get_drift_history_handler({"model_name": model_name})

            if isinstance(history_result, list) and history_result:
                history = history_result
                print(f"✅ Retrieved {len(history)} drift records for {model_name}")
            else:
                history = []
                print(f"⚠️ No drift history found for {model_name}")

        # Debug the data structure
        history = debug_drift_data(history)

        # Create chart
        chart = create_drift_chart(history)

        return history, chart

    except Exception as e:
        print(f"❌ Error refreshing drift history: {e}")
        import traceback
        traceback.print_exc()
        return [], gr.update(value=None)

def initialize_interface():
    """Initialize interface"""
    global current_model_mapping

    try:
        models = get_models_from_db()
        formatted_items, model_mapping = format_dropdown_items(models)
        current_model_mapping = model_mapping

        # Safe initialization
        if formatted_items:
            dropdown_value = formatted_items[0]
            first_model_name = extract_model_name_from_dropdown(dropdown_value, model_mapping)
            dropdown_update = gr.update(choices=formatted_items, value=dropdown_value)
        else:
            dropdown_value = None
            first_model_name = ""
            dropdown_update = gr.update(choices=[], value=None)

        return (
            dropdown_update,  # dropdown update
            "",  # new_model_name
            first_model_name,  # selected_model_display
            first_model_name  # drift_model_display
        )
    except Exception as e:
        print(f"❌ Error initializing interface: {e}")
        return (
            gr.update(choices=[], value=None),
            "",
            "",
            ""
        )


# --- Gradio Interface ---
def create_interface():
    """Create the Gradio interface"""
    with gr.Blocks(title="AI Model Management & Interaction Platform", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 AI Model Management & Interaction Platform")

        if not DATABASE_AVAILABLE:
            gr.Markdown("⚠️ **Demo Mode**: Running without database connectivity. Some features are simulated.")

        with gr.Row():
            # Left Column - Model Selection
            with gr.Column(scale=1):
                gr.Markdown("### 📋 Model Selection")

                model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Select Model",
                    interactive=True,
                    allow_custom_value=False,
                    value=None
                )

                search_box = gr.Textbox(
                    placeholder="Search by model name or description...",
                    label="🔍 Search Models"
                )

                create_new_button = gr.Button("➕ Create New Model", variant="secondary")

                # Create New Model Section
                with gr.Group(visible=False) as create_new_section:
                    gr.Markdown("#### 🆕 Create New Model")
                    new_model_name = gr.Textbox(
                        label="Model Name",
                        placeholder="Enter model name"
                    )
                    new_llm = gr.Dropdown(
                        choices=[
                            "gemini-1.0-pro",
                            "gemini-1.5-pro",
                            "groq-llama-3.1-8b-instant",
                            "groq-mixtral-8x7b-32768",
                            "claude-3-sonnet-20240229"
                        ],
                        label="Select LLM",
                        interactive=True
                    )
                    new_system_prompt = gr.Textbox(
                        label="System Prompt",
                        placeholder="Enter system prompt",
                        lines=3
                    )

                    with gr.Row():
                        enhance_button = gr.Button("✨ Enhance Prompt", variant="primary")
                        cancel_button = gr.Button("❌ Cancel", variant="secondary")

                    enhanced_prompt_display = gr.Textbox(
                        label="Enhanced Prompt",
                        interactive=False,
                        lines=4,
                        visible=False
                    )

                    prompt_choice = gr.Radio(
                        choices=["Keep Enhanced", "Keep Original"],
                        label="Choose Prompt",
                        visible=False
                    )

                    save_model_button = gr.Button("💾 Save Model", variant="primary", visible=False)
                    save_status = gr.Textbox(label="Status", interactive=False, visible=False)

            # Right Column - Model Operations
            with gr.Column(scale=2):
                gr.Markdown("### 🛠️ Model Operations")

                with gr.Tabs():
                    # Chatbot Tab
                    with gr.TabItem("💬 Chatbot"):
                        selected_model_display = gr.Textbox(
                            label="Currently Selected Model",
                            interactive=False
                        )

                        chatbot_interface = gr.Chatbot(
                            type="messages",
                            height=400,
                            show_label=False
                        )

                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Enter your message...",
                                label="Message",
                                scale=4
                            )
                            send_button = gr.Button("📤 Send", variant="primary", scale=1)

                        clear_chat = gr.Button("🗑️ Clear Chat", variant="secondary")

                    # Drift Analysis Tab
                    with gr.TabItem("📊 Drift Analysis"):
                        drift_model_display = gr.Textbox(
                            label="Model for Drift Analysis",
                            interactive=False
                        )

                        with gr.Row():
                            calculate_drift_button = gr.Button("🔍 Calculate New Drift", variant="primary")
                            refresh_history_button = gr.Button("🔄 Refresh History", variant="secondary")

                        drift_result = gr.Textbox(label="Latest Drift Calculation", interactive=False)

                        gr.Markdown("#### 📈 Drift History")
                        drift_history_display = gr.JSON(label="Drift History Data")

                        gr.Markdown("#### 📊 Drift Chart")
                        drift_chart = gr.Plot(label="Drift Over Time")

        # Event Handlers with better error handling
        search_box.change(update_model_dropdown, inputs=[search_box], outputs=[model_dropdown])
        model_dropdown.change(on_model_select, inputs=[model_dropdown],
                              outputs=[selected_model_display, drift_model_display])

        create_new_button.click(show_create_new, outputs=[create_new_section, new_model_name])
        cancel_button.click(cancel_create_new,
                            outputs=[create_new_section, new_model_name, new_system_prompt, enhanced_prompt_display,
                                     prompt_choice, save_model_button, save_status])

        enhance_button.click(enhance_prompt, inputs=[new_system_prompt],
                             outputs=[enhanced_prompt_display, prompt_choice, save_model_button])
        save_model_button.click(save_new_model,
                                inputs=[new_model_name, new_llm, new_system_prompt, enhanced_prompt_display,
                                        prompt_choice],
                                outputs=[save_status, save_status, model_dropdown])

        send_button.click(chatbot_response, inputs=[msg_input, chatbot_interface, model_dropdown],
                          outputs=[chatbot_interface, msg_input])
        msg_input.submit(chatbot_response, inputs=[msg_input, chatbot_interface, model_dropdown],
                         outputs=[chatbot_interface, msg_input])
        clear_chat.click(lambda: [], outputs=[chatbot_interface])

        calculate_drift_button.click(calculate_drift, inputs=[model_dropdown], outputs=[drift_result])
        refresh_history_button.click(refresh_drift_history, inputs=[model_dropdown],
                                     outputs=[drift_history_display, drift_chart])

        demo.load(initialize_interface,
                  outputs=[model_dropdown, new_model_name, selected_model_display, drift_model_display])

    return demo


def main():
    """Main function to launch the application"""
    print("🚀 Starting AI Model Management Platform...")

    # Create the interface
    demo = create_interface()

    # Launch configuration
    launch_config = {
        "server_name": "0.0.0.0",  # Listen on all interfaces
        "server_port": 7860,  # Default Gradio port
        "share": False,  # Set to True if you want a public link
        "show_error": True,  # Show detailed errors
        "quiet": False,  # Set to True to reduce output
        "show_api": True,  # Show API docs
    }

    print("📡 Launching Gradio interface...")
    print(f"🌐 Server will be available at:")
    print(f"   - Local: http://localhost:{launch_config['server_port']}")
    print(f"   - Network: http://0.0.0.0:{launch_config['server_port']}")

    try:
        demo.launch(**launch_config)
    except Exception as e:
        print(f"❌ Failed to launch Gradio interface: {e}")
        print("🔧 Troubleshooting suggestions:")
        print("   1. Check if port 7860 is already in use")
        print("   2. Try a different port: demo.launch(server_port=7861)")
        print("   3. Check firewall settings")
        print("   4. Ensure Gradio is properly installed: pip install gradio")
        return False

    return True


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Application error: {e}")
        sys.exit(1)