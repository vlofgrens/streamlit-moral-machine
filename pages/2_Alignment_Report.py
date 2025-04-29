import streamlit as st
import pandas as pd
import os

st.set_page_config(layout="wide")

# --- CSS for Bars and Layout ---
st.markdown("""
<style>
.alignment-grid {
    display: flex;
    flex-direction: column;
    gap: 20px;
}
.model-group, .indicator-group {
    /* border: 1px solid #ccc; */ /* Removed border */
    border-radius: 8px;
    padding: 15px;
    /* Remove background color */
    /* background-color: #f9f9f9; */ 
}
.model-group-title, .indicator-group-title {
    font-size: 1.5em;
    font-weight: bold;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #eee;
}
.indicator-grid {
    display: grid;
    /* Remove column definition - handled by dimension-row now */
    /* grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); */
    grid-template-columns: 1fr; /* Stack rows vertically */
    gap: 5px; /* Smaller gap between rows */
}

/* Adjust grid for better alignment, give explanation more defined space */
.dimension-row, 
/* REMOVE targeting direct children of indicator-grid for 4-col layout */
/* .indicator-grid > span, .indicator-grid > div */
.dimension-row /* Apply 4-col layout ONLY to dimension-row */
{
    display: grid;
    /* grid-template-columns: 150px 1fr 60px auto; */ /* Label | Bar | Score | Explanation */
    grid-template-columns: 150px 1fr 60px 2fr; /* Give explanation 2 fractions */
    gap: 10px;
    align-items: center;
    margin-bottom: 8px;
    padding: 5px;
    border-radius: 4px;
}

/* REMOVE rules specific to direct children of indicator-grid */
/* .indicator-grid > span.dimension-label, .indicator-grid > span.score-value, .indicator-grid > span.score-explanation {
     grid-column: span 1; 
}
.indicator-grid > .bar-container {
     grid-column: span 1;
} */

/* Style for indicator view to keep alignment - Now applied via dimension-row */
/* .indicator-grid .dimension-label { 
    font-weight: bold;
} */

.dimension-label {
    font-weight: 500;
    text-align: right;
    padding-right: 10px;
}

.bar-container {
    position: relative;
    height: 20px;
    /* background-color: #e9ecef; */ /* Original light gray */
    background-color: #495057; /* Dark Gray */
    border-radius: 4px;
    overflow: hidden; 
}

.bar {
    position: absolute;
    top: 0;
    height: 100%;
    border-radius: 4px;
    background-color: #000000;
}

.bar.positive {
    left: 50%; /* Start positive bar from the middle */
    /* background-color: #198754; */ /* Original Green */
    /* background-color: #00b300; */ /* More Saturated Green - Reverted */
    background-color: #28a745; /* Middle Saturated Green */
    /* background-color: #146c43; */ /* Darker Green - Reverted */
}

.bar.negative {
    right: 50%; /* Start negative bar from the middle, growing left */
    /* background-color: #dc3545; */ /* Original Red */
    /* background-color: #FF0000; */ /* More Saturated Red - Reverted */
    background-color: #e53935; /* Middle Saturated Red */
    /* background-color: #b02a37; */ /* Darker Red - Reverted */
}

.bar-zero-line {
    position: absolute;
    left: 50%;
    top: 0;
    bottom: 0;
    width: 1px;
    background-color: #adb5bd; 
    z-index: 1; /* Ensure line is visible */
}

.score-value {
    font-family: monospace;
    font-weight: bold;
    text-align: right;
    color: #C1CDCD;
}

.score-explanation {
    font-weight: bold;
    font-size: 1em;
    color: #FFFFFF;
    text-align: left;
}

/* View switcher styling */
.view-switcher a {
    display: inline-block;
    padding: 8px 15px;
    margin-right: 10px;
    border: 1px solid #ccc;
    border-radius: 5px;
    text-decoration: none;
    color: #333;
    background-color: #fff;
}
.view-switcher a.active {
    background-color: #0d6efd; /* Bootstrap primary blue */
    color: #fff;
    border-color: #0d6efd;
}

</style>
""", unsafe_allow_html=True)

# --- Helper Function for Explanation --- 
def get_explanation(indicator, score):
    if score is None:
        return "No score available"
    abs_score = abs(score)
    magnitude = ''
    if abs_score < 0.05:
        magnitude = 'Negligible'
    elif abs_score <= 0.25:
        magnitude = 'Slight'
    elif abs_score <= 0.75:
        magnitude = 'Moderate'
    else:
        magnitude = 'Strong'

    if magnitude == 'Negligible':
        return 'Negligible bias'

    explanation_map = {
        'Gender': ('sacrificing males', 'sacrificing females'),
        'Fitness': ('sacrificing healthy/beautiful', 'sacrificing unhealthy'),
        'Social value': ('sacrificing productive', 'sacrificing unproductive'),
        'Legal status': ('sacrificing law-abiding', 'sacrificing criminals'),
        'Relation to vehicle': ('sacrificing passengers', 'sacrificing pedestrians'),
        'Intervention': ('intervening', 'not intervening'),
        'Num humans': ('sacrificing larger groups', 'sacrificing smaller groups'),
        'Age': ('sacrificing elderly', 'sacrificing younger'),
        'Young preference': ('sacrificing young', 'aversion to sacrificing young'),
        'Adult preference': ('sacrificing adults', 'aversion to sacrificing adults'),
        'Elderly preference': ('sacrificing elderly', 'aversion to sacrificing elderly')
    }

    pref_positive, pref_negative = explanation_map.get(indicator, ('positive bias', 'negative bias'))
    
    direction = pref_positive if score > 0 else pref_negative
    return f"{magnitude} preference for {direction}"

# --- Page Content --- 
st.title("Model Alignment Report")
st.markdown("""
This report shows the quantified bias of different LLMs based on their decisions in various trolley problem scenarios.
Scores range from -1 to 1. The magnitude indicates the strength of the preference, and the sign indicates the direction based on the specific indicator.

[What do these indicators mean?](/Indicator_Explanations)
""") # Link to the other page

# --- CSV Data Loading and Processing --- 
# Determine the correct path to the CSV file relative to this script
# Assuming the script is in pages/ and csv is in streamlit_moral_machine/results/
script_dir = os.path.dirname(__file__)
# ADJUST path to look inside the 'results' subdirectory
# csv_path = os.path.join(script_dir, '..', 'v4_quant_result.csv') 
csv_path = os.path.join(script_dir, '..', 'results', 'v4_quant_result.csv') 

df = None
error_message = None
try:
    df = pd.read_csv(csv_path)

    # Data Cleaning 
    if not df.empty:
        model_col_original_name = df.columns[0]
        df = df.rename(columns={model_col_original_name: 'Model'})
        cleaned_columns = ['Model'] + [
            col.replace('_', ' ').capitalize() for col in df.columns[1:]
        ]
        df.columns = cleaned_columns
        indicators = cleaned_columns[1:] # Get indicator names after cleaning
        models = df['Model'].unique().tolist()
    else:
        error_message = "CSV file is empty."
        indicators = []
        models = []

except FileNotFoundError:
    # Update error message to reflect the new path expectation
    # error_message = f"Error: Alignment results file not found at `{csv_path}`. Make sure `v4_quant_result.csv` is in the `streamlit_moral_machine` directory."
    error_message = f"Error: Alignment results file not found at `{csv_path}`. Make sure `v4_quant_result.csv` is in the `streamlit_moral_machine/results` directory."
    indicators = []
    models = []
except Exception as e:
    error_message = f"Error reading or processing CSV file: {e}"
    indicators = []
    models = []

# --- View Switching --- 
# Use query params for view switching if preferred, or radio buttons
# default_view = st.query_params.get('group_by', 'model') # Query param method
default_view_index = 0 # Default to model view
grouping_mode = st.radio(
    "Group By:", 
    ('model', 'indicator'), 
    index=default_view_index, 
    horizontal=True,
    format_func=lambda x: "Model" if x == 'model' else "Indicator"
)

# --- Display Logic --- 
if error_message:
    st.error(error_message)
elif df is not None and not df.empty:
    st.markdown('<div class="alignment-grid">' , unsafe_allow_html=True)

    if grouping_mode == 'model':
        # --- Render Grouped by Model ---
        for model_name in models:
            model_data = df[df['Model'] == model_name].iloc[0] # Get row for the model
            st.markdown(f'<div class="model-group">'
                        f'<div class="model-group-title">{model_name}</div>', unsafe_allow_html=True)
            
            for indicator in indicators:
                score = model_data.get(indicator)
                # Handle potential non-numeric scores gracefully
                try:
                    score = float(score) if pd.notna(score) else 0.0
                except (ValueError, TypeError):
                    score = 0.0 # Default to 0 if conversion fails
                    
                explanation = get_explanation(indicator, score)
                bar_width_percent = min(abs(score) * 50, 50) # Cap at 50% for each side
                bar_class = 'positive' if score >= 0 else 'negative'
                
                # Use st.markdown to create the row with HTML/CSS
                st.markdown(f'''
                    <div class="dimension-row">
                        <span class="dimension-label">{indicator}:</span>
                        <div class="bar-container">
                            <div class="bar-zero-line"></div>
                            <div class="bar {bar_class}" style="width: {bar_width_percent}%;"></div>
                        </div>
                        <span class="score-value">{score:.3f}</span>
                        <span class="score-explanation">{explanation}</span>
                    </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True) # Close model-group

    elif grouping_mode == 'indicator':
         # --- Render Grouped by Indicator ---
        for indicator in indicators:
            st.markdown(f'<div class="indicator-group">'
                        f'<div class="indicator-group-title">{indicator}</div>'
                        f'<div class="indicator-grid">' # Start indicator grid
                        , unsafe_allow_html=True)
            
            # Iterate through each model for the current indicator
            for model_name in models:
                model_data = df[df['Model'] == model_name].iloc[0]
                score = model_data.get(indicator)
                 # Handle potential non-numeric scores gracefully
                try:
                    score = float(score) if pd.notna(score) else 0.0
                except (ValueError, TypeError):
                    score = 0.0 # Default to 0 if conversion fails

                explanation = get_explanation(indicator, score)
                bar_width_percent = min(abs(score) * 50, 50) # Cap at 50%
                bar_class = 'positive' if score >= 0 else 'negative'
                
                # WRAP the four elements in a single dimension-row div
                st.markdown(f'''
                    <div class="dimension-row"> 
                        <span class="dimension-label">{model_name}:</span>
                        <div class="bar-container">
                            <div class="bar-zero-line"></div>
                            <div class="bar {bar_class}" style="width: {bar_width_percent}%;"></div>
                        </div>
                        <span class="score-value">{score:.3f}</span>
                        <span class="score-explanation">{explanation}</span>
                    </div>
                ''', unsafe_allow_html=True)
            
            st.markdown('</div></div>', unsafe_allow_html=True) # Close indicator-grid and indicator-group

    st.markdown('</div>', unsafe_allow_html=True) # Close alignment-grid
elif df is not None and df.empty:
     st.warning("Alignment data loaded but the file is empty.")
# No else needed if error_message handles other cases 