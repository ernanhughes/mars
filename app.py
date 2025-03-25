import gradio as gr

from mars import analyze_ticker


def run_analysis(ticker):
    result = analyze_ticker(ticker)
    return f"""
### Plan
{result['plan']}

### Teacher Question
{result['teacher_question']}

### Critique
{result['critique']}

### Final Question
{result['final_question']}

### Signal
{result['signal']}

### Rationale
{result['rationale']}
"""


with gr.Blocks() as iface:
    gr.Markdown("# MARS Financial Reasoning")
    ticker_input = gr.Textbox(label="Enter stock ticker", placeholder="e.g., TSLA")
    run_button = gr.Button("Analyze", variant="primary")
    output_box = gr.Markdown()

    run_button.click(fn=run_analysis,
                     inputs=ticker_input,
                     outputs=output_box,
                     show_progress=True)  # <-- shows loading and disables button

if __name__ == "__main__":
    iface.launch()
