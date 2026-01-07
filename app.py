import gradio as gr
import sqlite3
import pandas as pd
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Find the TrackIO database
db_path = Path(".trackio")
db_files = list(db_path.glob("**/*.db"))

def get_training_data():
    """Extract and parse all training data from TrackIO database"""
    if not db_files:
        return None, None, "❌ No data found!"
    
    db_file = db_files[0]
    
    try:
        conn = sqlite3.connect(db_file)
        metrics_df = pd.read_sql_query("SELECT * FROM metrics ORDER BY timestamp", conn)
        
        # Parse the 'metrics' column (stored as bytes/JSON)
        parsed_data = []
        for _, row in metrics_df.iterrows():
            try:
                metrics_json = row['metrics']
                if isinstance(metrics_json, bytes):
                    metrics_json = metrics_json.decode('utf-8')
                
                metrics = json.loads(metrics_json)
                metrics['run_name'] = row['run_name']
                metrics['timestamp'] = row['timestamp']
                metrics['step'] = row['step']
                parsed_data.append(metrics)
            except:
                continue
        
        df = pd.DataFrame(parsed_data)
        
        # Separate Q8 and Q3 runs
        q8_df = df[df['run_name'].str.contains('q8', case=False)] if 'run_name' in df.columns else df
        q3_df = df[df['run_name'].str.contains('q3', case=False)] if 'run_name' in df.columns else pd.DataFrame()
        
        conn.close()
        
        message = f"✅ {len(q8_df)} Q8 entries"
        if not q3_df.empty:
            message += f" + {len(q3_df)} Q3 entries"
        
        return q8_df, q3_df, message
        
    except Exception as e:
        return None, None, f"❌ Error: {str(e)}"

def create_training_plot(q8_df, q3_df):
    """Create combined training plot for both Q8 and Q3"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Q8 Loss', 'Q8 F1 Score', 'Q3 Loss', 'Q3 F1 Score'),
        vertical_spacing=0.12,
        horizontal_spacing=0.1
    )
    
    # Q8 plots - filter to only rows with epoch data
    if q8_df is not None and not q8_df.empty:
        q8_plot_df = q8_df[q8_df['epoch'].notna()] if 'epoch' in q8_df.columns else q8_df
        
        if not q8_plot_df.empty and 'epoch' in q8_plot_df.columns:
            # Q8 Loss
            if 'train_loss' in q8_plot_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=q8_plot_df['epoch'], 
                        y=q8_plot_df['train_loss'],
                        mode='lines+markers', 
                        name='Q8 Loss',
                        line=dict(color='#EF4444', width=3),
                        marker=dict(size=7)
                    ),
                    row=1, col=1
                )
            
            # Q8 F1
            if 'val_f1' in q8_plot_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=q8_plot_df['epoch'], 
                        y=q8_plot_df['val_f1'],
                        mode='lines+markers', 
                        name='Q8 F1',
                        line=dict(color='#10B981', width=3),
                        marker=dict(size=7)
                    ),
                    row=1, col=2
                )
                
                # Mark Q8 peak
                best_idx = q8_plot_df['val_f1'].idxmax()
                fig.add_trace(
                    go.Scatter(
                        x=[q8_plot_df.loc[best_idx, 'epoch']], 
                        y=[q8_plot_df.loc[best_idx, 'val_f1']],
                        mode='markers', 
                        name='Q8 Peak',
                        marker=dict(size=16, color='#FBBF24', line=dict(width=2, color='#F59E0B')),
                        showlegend=False
                    ),
                    row=1, col=2
                )
    
    # Q3 plots
    if q3_df is not None and not q3_df.empty:
        q3_plot_df = q3_df[q3_df['epoch'].notna()] if 'epoch' in q3_df.columns else q3_df
        
        if not q3_plot_df.empty and 'epoch' in q3_plot_df.columns:
            # Q3 Loss
            if 'train_loss' in q3_plot_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=q3_plot_df['epoch'], 
                        y=q3_plot_df['train_loss'],
                        mode='lines+markers', 
                        name='Q3 Loss',
                        line=dict(color='#F59E0B', width=3),
                        marker=dict(size=7)
                    ),
                    row=2, col=1
                )
            
            # Q3 F1
            if 'val_f1' in q3_plot_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=q3_plot_df['epoch'], 
                        y=q3_plot_df['val_f1'],
                        mode='lines+markers', 
                        name='Q3 F1',
                        line=dict(color='#8B5CF6', width=3),
                        marker=dict(size=7)
                    ),
                    row=2, col=2
                )
                
                # Mark Q3 peak
                best_idx = q3_plot_df['val_f1'].idxmax()
                fig.add_trace(
                    go.Scatter(
                        x=[q3_plot_df.loc[best_idx, 'epoch']], 
                        y=[q3_plot_df.loc[best_idx, 'val_f1']],
                        mode='markers', 
                        name='Q3 Peak',
                        marker=dict(size=16, color='#FBBF24', line=dict(width=2, color='#F59E0B')),
                        showlegend=False
                    ),
                    row=2, col=2
                )
    
    fig.update_xaxes(title_text="Epoch", gridcolor='#374151', showgrid=True)
    fig.update_yaxes(gridcolor='#374151', showgrid=True)
    
    fig.update_layout(
        title_text="<b>Training Progress</b>",
        height=700,
        showlegend=True,
        template='plotly_dark',
        paper_bgcolor='#1F2937',
        plot_bgcolor='#111827',
        font=dict(size=12, color='#E5E7EB'),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor='rgba(31, 41, 55, 0.8)'
        )
    )
    
    return fig

def create_summary_cards(q8_df, q3_df):
    """Create summary cards"""
    q8_stats = ""
    
    if q8_df is not None and not q8_df.empty:
        # Get best F1 from ALL rows
        q8_best_f1 = 0
        if 'val_f1' in q8_df.columns:
            q8_best_f1 = float(q8_df['val_f1'].max())
        if 'best_val_f1' in q8_df.columns:
            best_from_summary = float(q8_df['best_val_f1'].max())
            q8_best_f1 = max(q8_best_f1, best_from_summary)
        
        # Other stats from epoch rows only
        epoch_rows = q8_df[q8_df['epoch'].notna()] if 'epoch' in q8_df.columns else q8_df
        q8_epochs = int(epoch_rows['epoch'].max()) if 'epoch' in epoch_rows.columns and not epoch_rows.empty else 0
        q8_final_loss = float(epoch_rows['train_loss'].iloc[-1]) if 'train_loss' in epoch_rows.columns and not epoch_rows.empty else 0
        
        q8_stats = f"""
        <div style="background: linear-gradient(135deg, #3B82F6 0%, #1E40AF 100%); padding: 30px; border-radius: 15px; color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
            <h2 style="margin: 0 0 20px 0; font-size: 22px; font-weight: 600;">Q8 Model</h2>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px">
                <div>
                    <div style="font-size: 13px; opacity: 0.85; margin-bottom: 5px;">Epochs</div>
                    <div style="font-size: 36px; font-weight: 700;">{q8_epochs}</div>
                </div>
                <div>
                    <div style="font-size: 13px; opacity: 0.85; margin-bottom: 5px;">Peak F1</div>
                    <div style="font-size: 36px; font-weight: 700;">{q8_best_f1:.3f}</div>
                </div>
                <div>
                    <div style="font-size: 13px; opacity: 0.85; margin-bottom: 5px;">Final Loss</div>
                    <div style="font-size: 36px; font-weight: 700;">{q8_final_loss:.3f}</div>
                </div>
            </div>
        </div>
        """
    
    q3_stats = ""
    if q3_df is not None and not q3_df.empty:
        q3_best_f1 = 0
        if 'val_f1' in q3_df.columns:
            q3_best_f1 = float(q3_df['val_f1'].max())
        if 'best_val_f1' in q3_df.columns:
            best_from_summary = float(q3_df['best_val_f1'].max())
            q3_best_f1 = max(q3_best_f1, best_from_summary)
        
        epoch_rows = q3_df[q3_df['epoch'].notna()] if 'epoch' in q3_df.columns else q3_df
        q3_epochs = int(epoch_rows['epoch'].max()) if 'epoch' in epoch_rows.columns and not epoch_rows.empty else 0
        q3_final_loss = float(epoch_rows['train_loss'].iloc[-1]) if 'train_loss' in epoch_rows.columns and not epoch_rows.empty else 0
        
        q3_stats = f"""
        <div style="background: linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%); padding: 30px; border-radius: 15px; color: white; box-shadow: 0 10px 30px rgba(0,0,0,0.5);">
            <h2 style="margin: 0 0 20px 0; font-size: 22px; font-weight: 600;">Q3 Model</h2>
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px">
                <div>
                    <div style="font-size: 13px; opacity: 0.85; margin-bottom: 5px;">Epochs</div>
                    <div style="font-size: 36px; font-weight: 700;">{q3_epochs}</div>
                </div>
                <div>
                    <div style="font-size: 13px; opacity: 0.85; margin-bottom: 5px;">Peak F1</div>
                    <div style="font-size: 36px; font-weight: 700;">{q3_best_f1:.3f}</div>
                </div>
                <div>
                    <div style="font-size: 13px; opacity: 0.85; margin-bottom: 5px;">Final Loss</div>
                    <div style="font-size: 36px; font-weight: 700;">{q3_final_loss:.3f}</div>
                </div>
            </div>
        </div>
        """
    
    html = f"""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 25px; margin: 25px 0;">
        {q8_stats}
        {q3_stats}
    </div>
    """
    
    return html

def show_dashboard():
    """Main dashboard function"""
    q8_df, q3_df, message = get_training_data()
    
    summary = f"""
    # 🧬 Protein Structure Prediction
    
    {message}
    """
    
    cards = create_summary_cards(q8_df, q3_df)
    plot = create_training_plot(q8_df, q3_df)
    
    return summary, cards, plot

# Gradio interface with dark theme
with gr.Blocks(title="Protein Structure Dashboard", theme=gr.themes.Base(
    primary_hue="blue",
    secondary_hue="purple",
).set(
    body_background_fill='#0F172A',
    body_background_fill_dark='#0F172A',
    background_fill_primary='#1E293B',
    background_fill_primary_dark='#1E293B',
    background_fill_secondary='#1E293B',
    background_fill_secondary_dark='#1E293B',
    block_background_fill='#1E293B',
    block_background_fill_dark='#1E293B',
    block_label_background_fill='#1E293B',
    block_label_background_fill_dark='#1E293B',
    input_background_fill='#334155',
    input_background_fill_dark='#334155',
    button_primary_background_fill='#3B82F6',
    button_primary_background_fill_dark='#3B82F6',
    block_title_text_color='#E5E7EB',
    block_label_text_color='#E5E7EB',
    body_text_color='#E5E7EB',
    body_text_color_subdued='#94A3B8',
)) as demo:
    
    gr.Markdown("""
    <div style="text-align: center; padding: 40px; background: linear-gradient(135deg, #1E3A8A 0%, #7C3AED 100%); border-radius: 20px; margin-bottom: 30px; box-shadow: 0 10px 40px rgba(0,0,0,0.4);">
        <h1 style="color: white; margin: 0; font-size: 32px; font-weight: 700;">Protein Secondary Structure</h1>
        <p style="color: rgba(255,255,255,0.9); margin: 15px 0 0 0; font-size: 16px;">BiLSTM + CNN</p>
    </div>
    """)
    
    with gr.Row():
        refresh_btn = gr.Button("🔄 Refresh", variant="primary", size="lg")
    
    summary_md = gr.Markdown()
    stats_cards = gr.HTML()
    main_plot = gr.Plot()
    
    # Load data on startup
    demo.load(
        fn=show_dashboard,
        outputs=[summary_md, stats_cards, main_plot]
    )
    
    # Refresh button
    refresh_btn.click(
        fn=show_dashboard,
        outputs=[summary_md, stats_cards, main_plot]
    )

if __name__ == "__main__":
    demo.launch()
