import gradio as gr
import os
from pathlib import Path

# Inside the Docker image the README is mounted at /app/README.md
os.environ["ENV_README_PATH"] = "/app/README.md"

from openenv.core.env_server.web_interface import create_web_interface_app
from src.env import BugTriageEnv
from src.models import BugTriageAction, BugTriageObservation

def format_bug_observation(data: dict) -> str:
    """Formats the BugTriageObservation for a premium Markdown display."""
    obs = data.get("observation", {})
    if not obs: return "### Click **New Bug** to start."
    
    report = obs.get("bug_report", {})
    task_id = obs.get("task_id", "Unknown Task")
    
    md = f"# 🐞 {report.get('title', 'No Title')}\n"
    md += f"**Repository:** `{report.get('repo', 'N/A')}` | **Current Task:** `{task_id}`\n\n"
    
    labels = report.get("labels", [])
    if labels:
        md += " ".join([f"`{l}`" for l in labels]) + "\n\n"
    
    md += "### 📄 Description\n"
    body = report.get('body', 'No description provided.')
    md += f"{body[:1200]}{'...' if len(body) > 1200 else ''}\n\n"
    
    if obs.get("available_assignees"):
        md += "#### 👥 Candidate Assignees\n"
        md += ", ".join([f"`{a}`" for a in obs["available_assignees"]]) + "\n"

    if data.get("done"):
        reward = data.get("reward", 0)
        md += f"\n---\n### ✅ Submission Result\n**Reward Score:** `{reward}`\n"
        info = data.get("info", {})
        if "ground_truth" in info:
            gt = info["ground_truth"]
            md += f"**Ground Truth:** `{gt.get('assignee', gt.get('criticality', gt.get('severity', '')))}`\n"
        
    return md

def custom_gradio_builder(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    """Builds a premium, documentation-first UI for judges."""
    readme_path = Path("/app/README.md")
    readme_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "README.md not found in /app/."

    with gr.Blocks(title="Dhurandhar Bug Triage") as demo:
        gr.Markdown("# 🤖 Bug Triage Dashboard")
        
        with gr.Row():
            with gr.Column(scale=3):
                obs_display = gr.Markdown("### Click **New Bug** to fetch a bug report from the dataset.")
                with gr.Accordion("Technical Payload (JSON)", open=False):
                    raw_json = gr.JSON()
            
            with gr.Column(scale=2):
                gr.Markdown("### 🎯 Environmental Step")
                with gr.Group():
                    inputs = []
                    for f in action_fields:
                        label = f["name"].replace("_", " ").title()
                        if f.get("type") == "select" or "choices" in f:
                            inputs.append(gr.Dropdown(choices=f.get("choices", []), label=label, interactive=True))
                        elif f["name"] == "reasoning":
                            inputs.append(gr.Textbox(label=label, lines=3, placeholder="AI Reasoning..."))
                        elif f["name"] == "confidence":
                            inputs.append(gr.Slider(0, 1, value=0.8, step=0.1, label=label))
                        else:
                            # Hide task_id and bug_id as they are contextual
                            visible = f["name"] not in ["task_id", "bug_id"]
                            inputs.append(gr.Textbox(label=label, visible=visible))
                
                with gr.Row():
                    reset_btn = gr.Button("🔄 New Bug", variant="secondary", size="lg")
                    step_btn = gr.Button("🚀 Submit Decision", variant="primary", size="lg")
                
                gr.Markdown("> **Goal:** Correct 500+ real-world bugs across 15 repositories.")

        # Event Handlers
        async def run_reset():
            data = await web_manager.reset_environment()
            return format_bug_observation(data), data, data["observation"].get("task_id", ""), data["observation"]["bug_report"].get("bug_id", "")
        
        async def run_step(*vals):
            # Map values back to field names
            action_data = {}
            for i, f in enumerate(action_fields):
                # Filter out None or empty strings from Gradio UI
                if vals[i] is not None and vals[i] != "":
                    action_data[f["name"]] = vals[i]
            
            data = await web_manager.step_environment(action_data)
            return format_bug_observation(data), data

        # Wire reset() to also populate the hidden task_id and bug_id fields
        tid_idx = next(i for i, f in enumerate(action_fields) if f["name"] == "task_id")
        bid_idx = next(i for i, f in enumerate(action_fields) if f["name"] == "bug_id")
        
        reset_btn.click(run_reset, outputs=[obs_display, raw_json, inputs[tid_idx], inputs[bid_idx]])
        step_btn.click(run_step, inputs=inputs, outputs=[obs_display, raw_json])
        
    return demo

from openenv.core.env_server.web_interface import WebInterfaceManager, load_environment_metadata, _extract_action_fields, _is_chat_env, get_quick_start_markdown, build_gradio_app
from openenv.core.env_server.http_server import create_fastapi_app
from openenv.core.env_server.gradio_theme import OPENENV_GRADIO_CSS, OPENENV_GRADIO_THEME
from fastapi.responses import RedirectResponse

def create_env():
    return BugTriageEnv(data_path="data/bugs_processed.json", task_type="all", seed=42)

# 1. Initialize Metadata & Manager
metadata = load_environment_metadata(create_env, env_name="bug-triage")
web_manager = WebInterfaceManager(create_env, BugTriageAction, BugTriageObservation, metadata)
action_fields = _extract_action_fields(BugTriageAction)


def build_custom_playground(web_manager, action_fields):
    """A clean, full-width implementation of the technical console."""
    with gr.Column():
        gr.Markdown("### 🛠️ Developer Console")
        gr.Markdown("Directly interact with the environment via the API. No sidebars, 100% width.")
        
        with gr.Row():
            with gr.Column(scale=3):
                obs_display = gr.Markdown("Click **Reset** to begin.")
                with gr.Accordion("Raw Observation (JSON)", open=False):
                    raw_json = gr.JSON()
            
            with gr.Column(scale=2):
                inputs = []
                for f in action_fields:
                    label = f["name"].replace("_", " ").title()
                    if f.get("type") == "select" or "choices" in f:
                        inputs.append(gr.Dropdown(choices=f.get("choices", []), label=label))
                    elif f["name"] == "reasoning":
                        inputs.append(gr.Textbox(label=label, lines=3))
                    elif f["name"] == "confidence":
                        inputs.append(gr.Slider(0, 1, value=0.8, step=0.1, label=label))
                    else:
                        inputs.append(gr.Textbox(label=label))
                
                with gr.Row():
                    step_btn = gr.Button("Step", variant="primary")
                    reset_btn = gr.Button("Reset", variant="secondary")
                
                status = gr.Textbox(label="Status", interactive=False)

    async def run_reset():
        data = await web_manager.reset_environment()
        return "New environment instance created.", data, data["observation"]
        
    async def run_step(*vals):
        action = {f["name"]: vals[i] for i, f in enumerate(action_fields)}
        res = await web_manager.step_environment(action)
        return f"Step complete. Reward: {res.get('reward', 0)}", res, res["observation"]

    reset_btn.click(run_reset, outputs=[status, raw_json, obs_display])
    step_btn.click(run_step, inputs=inputs, outputs=[status, raw_json, obs_display])

def build_final_ui():
    readme_path = Path("/app/README.md")
    readme_content = readme_path.read_text(encoding="utf-8") if readme_path.exists() else "README.md not found in /app/."
    
    # Create clean metadata to HIDE all sidebars in other tabs
    metadata_clean = metadata.model_copy()
    metadata_clean.readme_content = None
    
    # Aggressive CSS to kill all sidebars and preserve true full-width
    CUSTOM_FULL_WIDTH_CSS = OPENENV_GRADIO_CSS + """
    .gradio-container { max-width: 100% !important; margin: 0 !important; width: 100% !important; }
    .col-left { display: none !important; width: 0 !important; visibility: hidden !important; }
    .col-right { width: 100% !important; flex: 1 !important; }
    .full-width-doc { max-width: 100% !important; width: 100% !important; padding: 40px !important; }
    """

    # Aggressive JS to physically DELETE the sidebar from the browser DOM
    SIDEBAR_KILLER_JS = """
    function() {
        const kill = () => {
            document.querySelectorAll('.col-left').forEach(el => el.remove());
            document.querySelectorAll('.row').forEach(r => { 
                if(r.children.length === 1) r.children[0].style.width = '100%'; 
            });
        };
        kill();
        setInterval(kill, 500); // Check every 500ms and kill it if it reappears
    }
    """
    
    with gr.Blocks(theme=OPENENV_GRADIO_THEME, css=CUSTOM_FULL_WIDTH_CSS, js=SIDEBAR_KILLER_JS) as demo:
        with gr.Tabs() as tabs:
            # 1. THE HERO TAB (Truly Full Screen Documentation)
            with gr.Tab("📖 Project Documentation", id="readme"):
                gr.Markdown(readme_content, elem_classes="full-width-doc")
            
            # 2. THE JUDGE DASHBOARD (Premium Custom View)
            with gr.Tab("🚀 Judge Dashboard", id="custom"):
                custom_gradio_builder(web_manager, action_fields, metadata_clean, False, "Dhurandhar", None)
                
            # 3. THE TECHNICAL PLAYGROUND (Custom Full-Width View)
            with gr.Tab("🛠️ Technical Playground", id="playground"):
                build_custom_playground(web_manager, action_fields)
                
    return demo

# 3. Build the FastAPI app (registers /reset, /step, /health, /schema, etc.)
app = create_fastapi_app(create_env, BugTriageAction, BugTriageObservation)

# Mount Gradio directly at / (the root path) to avoid redirect issues in HuggingFace spaces
app = gr.mount_gradio_app(app, build_final_ui(), path="/")

def main():
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 7860))
    print(f"🚀 Starting Dhurandhar Bug Triage Environment on port {port}...")
    print(f"🔗 UI available at: http://0.0.0.0:{port}/")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
