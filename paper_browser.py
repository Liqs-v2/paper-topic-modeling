import pandas as pd
import gradio as gr

# Load the CSV file
def load_papers():
    """Load the papers dataset from CSV file and join with document info"""
    # Load the main papers dataset
    papers_df = pd.read_csv('papers_df.csv')
    
    # Load the document info dataset
    document_info_df = pd.read_csv('gpt_41_mini-document_info.csv')
    
    # Join the dataframes on 'document' (papers_df) and 'Document' (document_info_df)
    merged_df = papers_df.merge(document_info_df, left_on='document', right_on='Document', how='left')
    
    return merged_df

# Initialize the dataset
full_papers_df = load_papers()
papers_df = full_papers_df.copy()  # This will be the filtered dataset
current_index = 0

# Get unique topic names for the dropdown
topic_names = sorted(full_papers_df['Name'].dropna().unique().tolist())

def filter_papers_by_topic(selected_topic):
    """Filter papers based on selected topic"""
    global papers_df, current_index
    
    if not selected_topic or selected_topic == "All Topics":
        # If no topic selected or "All Topics" selected, show all papers
        papers_df = full_papers_df.copy()
    else:
        # Filter papers by selected topic
        papers_df = full_papers_df[full_papers_df['Name'] == selected_topic].copy()
    
    # Reset index to 0 when filtering changes
    current_index = 0
    
    # Update the jump input maximum
    return len(papers_df)

def get_paper_info(index):
    """Get paper information for the given index"""
    if index < 0 or index >= len(papers_df):
        return "Invalid index", "", "", "", f"Paper {index + 1} of {len(papers_df)}"
    
    paper = papers_df.iloc[index]
    title = paper['title'] if pd.notna(paper['title']) else "No title available"
    authors = paper['authors'] if pd.notna(paper['authors']) else "No authors listed"
    abstract = paper['abstract'] if pd.notna(paper['abstract']) else "No abstract available"
    topic = paper['Name'] if pd.notna(paper['Name']) else "No topic assigned"
    
    # Clean up authors field (remove brackets and quotes if present)
    if authors.startswith('[') and authors.endswith(']'):
        authors = authors[1:-1].replace("'", "").replace('"', '')
    
    paper_counter = f"Paper {index + 1} of {len(papers_df)}"
    
    return title, authors, abstract, topic, paper_counter

def go_to_paper(index):
    """Navigate to a specific paper by index"""
    global current_index
    if 0 <= index < len(papers_df):
        current_index = index
        return get_paper_info(current_index)
    return get_paper_info(current_index)

def next_paper():
    """Go to the next paper"""
    global current_index
    if current_index < len(papers_df) - 1:
        current_index += 1
    return get_paper_info(current_index)

def prev_paper():
    """Go to the previous paper"""
    global current_index
    if current_index > 0:
        current_index -= 1
    return get_paper_info(current_index)

def first_paper():
    """Go to the first paper"""
    global current_index
    current_index = 0
    return get_paper_info(current_index)

def last_paper():
    """Go to the last paper"""
    global current_index
    current_index = len(papers_df) - 1
    return get_paper_info(current_index)

def jump_to_paper(paper_number):
    """Jump to a specific paper number (1-indexed)"""
    global current_index
    if 1 <= paper_number <= len(papers_df):
        current_index = paper_number - 1
        return get_paper_info(current_index)
    return get_paper_info(current_index)

def on_topic_change(selected_topic):
    """Handle topic selection changes"""
    max_papers = filter_papers_by_topic(selected_topic)
    # Return updated values for all outputs
    title, authors, abstract, topic, paper_counter = get_paper_info(current_index)
    return title, authors, abstract, topic, paper_counter, gr.update(maximum=max_papers, value=1)

# Create custom theme with better font
custom_theme = gr.themes.Soft(
    font=["Arial", "Helvetica", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "Consolas", "monospace"]
)

# Create the Gradio interface
with gr.Blocks(title="Paper Browser", theme=custom_theme) as demo:
    gr.Markdown("# 📚 Paper Browser")
    gr.Markdown(f"Browse through {len(full_papers_df)} papers from the ACL 2025 dataset")
    
    # Topic filter
    topic_filter = gr.Dropdown(
        label="Filter by Topic",
        choices=["All Topics"] + topic_names,
        value="All Topics",
        interactive=True
    )
    
    # Paper counter
    paper_counter = gr.Textbox(
        label="Current Position",
        value=f"Paper 1 of {len(papers_df)}",
        interactive=False
    )
    
    # Navigation controls
    with gr.Row():
        first_btn = gr.Button("⏮️ First", variant="secondary")
        prev_btn = gr.Button("⬅️ Previous", variant="secondary")
        next_btn = gr.Button("➡️ Next", variant="secondary")
        last_btn = gr.Button("⏭️ Last", variant="secondary")
    
    # Jump to specific paper
    with gr.Row():
        jump_input = gr.Number(
            label="Jump to Paper Number",
            value=1,
            minimum=1,
            maximum=len(papers_df),
            step=1,
            precision=0
        )
        jump_btn = gr.Button("🔍 Jump", variant="primary")
    
    # Paper information display
    title_box = gr.Textbox(
        label="Title",
        value=papers_df.iloc[0]['title'],
        lines=2,
        interactive=False
    )
    
    authors_box = gr.Textbox(
        label="Authors",
        value=papers_df.iloc[0]['authors'],
        lines=3,
        interactive=False
    )
    
    topic_box = gr.Textbox(
        label="Topic",
        value=papers_df.iloc[0]['Name'] if pd.notna(papers_df.iloc[0]['Name']) else "No topic assigned",
        lines=1,
        interactive=False
    )
    
    abstract_box = gr.Textbox(
        label="Abstract",
        value=papers_df.iloc[0]['abstract'] if pd.notna(papers_df.iloc[0]['abstract']) else "No abstract available",
        lines=10,
        interactive=False
    )
    
    # Connect topic filter to update function
    topic_filter.change(
        on_topic_change,
        inputs=[topic_filter],
        outputs=[title_box, authors_box, abstract_box, topic_box, paper_counter, jump_input]
    )
    
    # Connect buttons to functions
    first_btn.click(
        first_paper,
        outputs=[title_box, authors_box, abstract_box, topic_box, paper_counter]
    )
    
    prev_btn.click(
        prev_paper,
        outputs=[title_box, authors_box, abstract_box, topic_box, paper_counter]
    )
    
    next_btn.click(
        next_paper,
        outputs=[title_box, authors_box, abstract_box, topic_box, paper_counter]
    )
    
    last_btn.click(
        last_paper,
        outputs=[title_box, authors_box, abstract_box, topic_box, paper_counter]
    )
    
    jump_btn.click(
        jump_to_paper,
        inputs=[jump_input],
        outputs=[title_box, authors_box, abstract_box, topic_box, paper_counter]
    )
    
    # Add some statistics
    gr.Markdown("---")
    gr.Markdown(f"**Dataset Statistics:**")
    gr.Markdown(f"- Total papers: {len(full_papers_df)}")
    gr.Markdown(f"- Papers with abstracts: {full_papers_df['abstract'].notna().sum()}")
    gr.Markdown(f"- Papers without abstracts: {full_papers_df['abstract'].isna().sum()}")
    gr.Markdown(f"- Unique topics: {len(topic_names)}")

if __name__ == "__main__":
    demo.launch(share=True) 