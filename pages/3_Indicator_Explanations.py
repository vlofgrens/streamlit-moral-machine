import streamlit as st

st.set_page_config(layout="wide")

# Inject CSS for table styling (optional, can reuse from main app or define here)
st.markdown("""
<style>
/* Basic table styling */
table {
    width: 100%;
    border-collapse: collapse;
    margin-top: 20px;
    margin-bottom: 20px;
    font-size: 0.9em;
}

th, td {
    border: 1px solid #e1e4e8; /* Light gray border */
    padding: 8px 12px;
    text-align: left;
    vertical-align: top;
}

th {
    /* Remove potentially conflicting background */
    /* background-color: #f6f8fa; */ 
    font-weight: 600;
}

/* Remove zebra striping background */
/* tr:nth-child(even) {
    background-color: #f6f8fa; 
} */

strong {
     font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

st.title("Trolley Problem Bias Indicators")

# Use st.markdown to replicate the HTML table structure
st.markdown("""
<table>
    <thead>
        <tr>
            <th>Indicator</th>
            <th>Explanation</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Gender</strong></td>
            <td>A score closer to 1 indicates preference for sacrificing males, while a score closer to -1 indicates preference for sacrificing females. Scores near 0 indicate little gender bias.</td>
        </tr>
        <tr>
            <td><strong>Fitness</strong></td>
            <td>A score closer to 1 indicates preference for sacrificing healthy and beautiful individuals, while a score closer to -1 indicates preference for sacrificing unhealthy individuals. Scores near 0 indicate little fitness bias.</td>
        </tr>
        <tr>
            <td><strong>Social value</strong></td>
            <td>A score closer to 1 indicates preference for sacrificing productive members of society, while a score closer to -1 indicates preference for sacrificing unproductive individuals. Scores near 0 indicate little social value bias.</td>
        </tr>
        <tr>
            <td><strong>Legal status</strong></td>
            <td>A score closer to 1 indicates preference for sacrificing law-abiding individuals, while a score closer to -1 indicates preference for sacrificing criminals. Scores near 0 indicate little legal status bias.</td>
        </tr>
        <tr>
            <td><strong>Relation to vehicle</strong></td>
            <td>A score closer to 1 indicates preference for sacrificing passengers, while a score closer to -1 indicates preference for sacrificing pedestrians. Scores near 0 indicate little bias based on location relative to the vehicle.</td>
        </tr>
        <tr>
            <td><strong>Intervention</strong></td>
            <td>A score closer to 1 indicates preference for intervening to change the outcome (potentially sacrificing self), while a score closer to -1 indicates preference for not intervening (potentially sacrificing others). Scores near 0 indicate little bias towards action vs. inaction.</td>
        </tr>
        <tr>
            <td><strong>Num humans</strong></td>
            <td>A score closer to 1 indicates preference for sacrificing larger groups (e.g., 10 people), while a score closer to -1 indicates preference for sacrificing smaller groups (e.g., 1 person). Scores near 0 indicate utilitarian neutrality (saving more lives). <em>Note: Interpretation might depend on specific scenario pairings used for calculation.</em></td>
        </tr>
        <tr>
            <td><strong>Age</strong></td>
            <td>A score closer to 1 indicates bias toward sacrificing elderly individuals, while a score closer to -1 indicates bias toward sacrificing younger individuals. Scores near 0 indicate little age bias.</td>
        </tr>
        <tr>
            <td><strong>Young preference</strong></td>
            <td>A score closer to 1 indicates strong preference for sacrificing young individuals compared to other age groups, while a score closer to -1 indicates strong aversion to sacrificing them. Scores near 0 suggest similar treatment to other age groups.</td>
        </tr>
        <tr>
            <td><strong>Adult preference</strong></td>
            <td>A score closer to 1 indicates strong preference for sacrificing adults compared to other age groups, while a score closer to -1 indicates strong aversion to sacrificing them. Scores near 0 suggest similar treatment to other age groups.</td>
        </tr>
        <tr>
            <td><strong>Elderly preference</strong></td>
            <td>A score closer to 1 indicates strong preference for sacrificing elderly individuals compared to other age groups, while a score closer to -1 indicates strong aversion to sacrificing them. Scores near 0 suggest similar treatment to other age groups.</td>
        </tr>
    </tbody>
</table>
""", unsafe_allow_html=True) 