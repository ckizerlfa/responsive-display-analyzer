import streamlit as st
import pandas as pd
from io import BytesIO
import chardet
import openpyxl
from PIL import Image
import requests
import re

# Set a constant for image width (slightly larger than before)
IMAGE_WIDTH = 150  # increased from 120 to 150

# Configure the Streamlit page
st.set_page_config(page_title="Responsive Display Asset Analysis Dashboard", layout="wide")

# Main Title & Intro
st.title("Responsive Display Asset Analysis Dashboard")
st.markdown(
    """
This dashboard aggregates and analyzes the performance of your responsive display assets across multiple campaigns.
Upload your CSV reports, filter your data, and receive actionable insights to optimize your creative assets.
    """
)

# -------------------------------
# Helper Functions
# -------------------------------

def detect_encoding(file):
    """Detect the encoding of a file using chardet."""
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    file.seek(0)  # Reset file pointer after reading
    return encoding

def generate_left_cell_html(asset, asset_type):
    """
    Generate HTML for the left cell of a recommendation row.
    For 'headline' or 'description', return the text.
    For images, return an <img> tag with a width of IMAGE_WIDTH.
    For YouTube videos, return an <iframe> with width IMAGE_WIDTH.
    Otherwise, return a clickable link.
    """
    asset_type_lower = asset_type.lower()
    if asset_type_lower in ["headline", "description"]:
        return f"{asset}"
    elif asset_type_lower in ['square image', 'landscape image', 'logo', 'landscape logo']:
        if re.match(r'^https?://', asset):
            return f"<img src='{asset}' style='width:{IMAGE_WIDTH}px;'>"
        else:
            return asset.split('/')[-1]
    elif asset_type_lower == "youtube video":
        youtube_match = re.search(r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)', asset)
        if youtube_match:
            video_id = youtube_match.group(1)
            # Using a 16:9 aspect ratio: width IMAGE_WIDTH, height roughly IMAGE_WIDTH * 9/16
            return f"<iframe width='{IMAGE_WIDTH}' height='{int(IMAGE_WIDTH * 9/16)}' src='https://www.youtube.com/embed/{video_id}' frameborder='0' allowfullscreen></iframe>"
        else:
            return f"<a href='{asset}'>View Asset</a>"
    else:
        if re.search(r'\.(png|jpg|jpeg|gif)$', asset, re.IGNORECASE):
            return f"<img src='{asset}' style='width:{IMAGE_WIDTH}px;'>"
        else:
            return f"<a href='{asset}'>View Asset</a>"

def display_asset_preview(asset_url, asset_type):
    """
    Display an asset preview using Streamlit commands.
    For image types, attempt to show an image;
    for YouTube videos, embed the video;
    otherwise, if the URL looks like an image, try to load it, else show a clickable link.
    This function is used in both the Detailed Asset View and the Actionable Recommendations.
    """
    asset_type_lower = asset_type.lower()
    if asset_type_lower in ['square image', 'landscape image', 'logo', 'landscape logo']:
        if re.match(r'^https?://', asset_url):
            try:
                response = requests.get(asset_url, timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=IMAGE_WIDTH)
                else:
                    st.markdown(f"[View Asset]({asset_url})")
            except requests.exceptions.Timeout:
                st.write("Request timed out.")
            except Exception as e:
                st.write(f"Error loading image: {e}")
        else:
            st.write(asset_url.split('/')[-1])
    elif asset_type_lower == 'youtube video':
        youtube_match = re.search(
            r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)',
            asset_url,
        )
        if youtube_match:
            video_id = youtube_match.group(1)
            st.video(f"https://www.youtube.com/embed/{video_id}")
        else:
            st.markdown(f"[View Asset]({asset_url})")
    else:
        if re.search(r'\.(png|jpg|jpeg|gif)$', asset_url, re.IGNORECASE):
            try:
                response = requests.get(asset_url, timeout=5)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    st.image(img, width=IMAGE_WIDTH)
                else:
                    st.markdown(f"[View Asset]({asset_url})")
            except Exception:
                st.markdown(f"[View Asset]({asset_url})")
        else:
            st.markdown(f"[View Asset]({asset_url})")

@st.cache_data(show_spinner=True)
def process_uploaded_files(uploaded_files):
    """Process uploaded CSV files and return a combined DataFrame."""
    combined_data = []
    line1, line2, header_line = None, None, None

    for idx, file in enumerate(uploaded_files):
        encoding = detect_encoding(file)
        try:
            content = file.getvalue().decode(encoding)
        except UnicodeDecodeError:
            st.error(f"Could not decode file {file.name}. Please check its encoding.")
            continue

        lines = content.splitlines()
        if idx == 0:
            line1 = lines[0]
            line2 = lines[1]
            header_line = lines[2].strip().split('\t')
            data_lines = lines[3:]
        else:
            data_lines = lines[3:]

        data = [line.split('\t') for line in data_lines if line.strip()]
        if not data:
            continue
        df = pd.DataFrame(data, columns=header_line)
        df['filename'] = file.name
        combined_data.append(df)

    if not combined_data:
        return None

    combined_df = pd.concat(combined_data, ignore_index=True)
    combined_df.columns = combined_df.columns.str.lower()

    if 'asset status' in combined_df.columns:
        combined_df = combined_df[combined_df['asset status'].str.lower() != 'removed']

    return combined_df

# -------------------------------
# Sidebar: File Upload & Process Button
# -------------------------------

st.sidebar.header("1. Upload Your CSV Reports")
uploaded_files = st.sidebar.file_uploader("Select CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    process_button = st.sidebar.button("Process Files")
else:
    st.sidebar.info("Please upload at least one CSV file.")

# -------------------------------
# Main Processing Block
# -------------------------------

if uploaded_files and process_button:
    with st.spinner("Processing files..."):
        combined_df = process_uploaded_files(uploaded_files)

    if combined_df is None:
        st.error("No valid data could be processed from the uploaded files.")
    else:
        required_columns = ['asset status', 'asset', 'status', 'asset type', 'performance']
        missing_cols = [col for col in required_columns if col not in combined_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
        else:
            combined_df['performance'] = combined_df['performance'].str.title()

            st.sidebar.header("2. Filter Your Data")
            asset_types = combined_df['asset type'].unique().tolist()
            selected_asset_types = st.sidebar.multiselect("Select Asset Types", options=asset_types, default=asset_types)

            campaigns = combined_df['filename'].unique().tolist()
            selected_campaigns = st.sidebar.multiselect("Select Campaigns (by filename)", options=campaigns, default=campaigns)

            filtered_df = combined_df[
                (combined_df['asset type'].isin(selected_asset_types)) &
                (combined_df['filename'].isin(selected_campaigns))
            ]

            # -------------------------------
            # Top Table: Asset Performance Summary aggregated by Asset Type
            # -------------------------------
            allowed_perf_columns = ["Best", "Good", "Low"]

            performance_summary_type = filtered_df.pivot_table(
                index='asset type',
                columns='performance',
                aggfunc='size',
                fill_value=0,
            ).reset_index()
            performance_summary_type.columns.name = None

            # Ensure the allowed columns exist
            for col in allowed_perf_columns:
                if col not in performance_summary_type.columns:
                    performance_summary_type[col] = 0

            performance_summary_type['Total'] = performance_summary_type[allowed_perf_columns].sum(axis=1)
            display_columns = ['asset type'] + allowed_perf_columns + ['Total']
            performance_summary_type = performance_summary_type[display_columns]

            def highlight_performance(row):
                # Highlight the largest number for each row
                max_val = row.max()
                styles = []
                for col, value in row.items():
                    if value == max_val and max_val > 0:
                        if col == 'Best':
                            styles.append('background-color: green; color: white;')
                        elif col == 'Good':
                            styles.append('background-color: blue; color: white;')
                        elif col == 'Low':
                            styles.append('background-color: red; color: white;')
                        else:
                            styles.append('')
                    else:
                        styles.append('')
                return styles

            sorted_perf_type = performance_summary_type.sort_values(by='Total', ascending=False)
            styled_perf_type = sorted_perf_type.style.apply(highlight_performance, subset=allowed_perf_columns, axis=1)
            styled_perf_type = styled_perf_type.applymap(lambda v: 'color: darkgrey;' if v == 0 else '', subset=allowed_perf_columns)

            st.subheader("Asset Performance Summary (by Asset Type)")
            st.dataframe(styled_perf_type, height=400)

            # -------------------------------
            # Create Performance Summary Pivot Table aggregated by Individual Assets
            # (For the tables below: actionable recommendations, detailed asset view, and download)
            # -------------------------------
            performance_summary_individual = filtered_df.pivot_table(
                index=['asset', 'asset type'],
                columns='performance',
                aggfunc='size',
                fill_value=0,
            ).reset_index()
            performance_summary_individual.columns.name = None

            for col in allowed_perf_columns:
                if col not in performance_summary_individual.columns:
                    performance_summary_individual[col] = 0

            performance_summary_individual['Total'] = performance_summary_individual[allowed_perf_columns].sum(axis=1)
            display_columns = ['asset', 'asset type'] + allowed_perf_columns + ['Total']
            performance_summary_individual = performance_summary_individual[display_columns]

            sorted_perf_individual = performance_summary_individual.sort_values(by='Total', ascending=False)

            # -------------------------------
            # Scorecard: Percentage of all Ranked Assets with Best, Good, and Low
            # -------------------------------
            # Consider only ranked assets (i.e. those with a performance in allowed_perf_columns)
            ranked_df = filtered_df[filtered_df['performance'].isin(allowed_perf_columns)]
            total_ranked = ranked_df.shape[0]
            best_pct = (ranked_df[ranked_df['performance'] == 'Best'].shape[0] / total_ranked * 100) if total_ranked > 0 else 0
            good_pct = (ranked_df[ranked_df['performance'] == 'Good'].shape[0] / total_ranked * 100) if total_ranked > 0 else 0
            low_pct = (ranked_df[ranked_df['performance'] == 'Low'].shape[0] / total_ranked * 100) if total_ranked > 0 else 0

            st.subheader("Overall Ranked Asset Scorecard")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Best (%)", value=f"{best_pct:.1f}%")
            with col2:
                st.metric(label="Good (%)", value=f"{good_pct:.1f}%")
            with col3:
                st.metric(label="Low (%)", value=f"{low_pct:.1f}%")

            # -------------------------------
            # Actionable Recommendations (Based on Individual Assets)
            # -------------------------------
            recommendations_dict = {"Best": [], "Good": [], "Low": []}
            for idx, row in sorted_perf_individual.iterrows():
                for perf in allowed_perf_columns:
                    if row['Total'] > 0 and row.get(perf, 0) > 0.5 * row['Total']:
                        recommendations_dict[perf].append((row, perf))
                        break

            st.subheader("Actionable Recommendations")
            for perf in ["Best", "Good", "Low"]:
                if recommendations_dict[perf]:
                    st.markdown(f"#### Majority {perf}")
                    for row, perf_label in recommendations_dict[perf]:
                        col1, col2 = st.columns([1.5, 3])
                        with col1:
                            asset_type_lower = row["asset type"].lower()
                            if asset_type_lower in ["headline", "description"]:
                                st.markdown(f"{row['asset']}")
                            elif asset_type_lower == "youtube video":
                                display_asset_preview(row['asset'], row['asset type'])
                            else:
                                st.markdown(generate_left_cell_html(row['asset'], row['asset type']), unsafe_allow_html=True)
                        with col2:
                            st.markdown(
                                f"**Asset Type:** {row['asset type']}  \n"
                                f"**Performance:** Best: {row.get('Best', 0)}, Good: {row.get('Good', 0)}, Low: {row.get('Low', 0)}"
                            )
                        st.markdown("<hr>", unsafe_allow_html=True)
                else:
                    st.info(f"No assets with majority {perf} performance.")

            # -------------------------------
            # Detailed Asset View (Based on Individual Assets)
            # -------------------------------
            st.subheader("Detailed Asset View")
            for asset_type in selected_asset_types:
                st.markdown(f"### {asset_type.title()}")
                asset_subset = sorted_perf_individual[sorted_perf_individual['asset type'] == asset_type]
                for idx, row in asset_subset.iterrows():
                    st.markdown(f"**Asset:** {row['asset']}")
                    cols = st.columns([3, 1, 1, 1])
                    asset_url = row['asset']
                    with cols[0]:
                        display_asset_preview(asset_url, row['asset type'])
                    with cols[1]:
                        st.write(f"Best: {row.get('Best', 0)}")
                    with cols[2]:
                        st.write(f"Good: {row.get('Good', 0)}")
                    with cols[3]:
                        st.write(f"Low: {row.get('Low', 0)}")
                    st.markdown("---")

            # -------------------------------
            # Download Option for the Individual Asset Summary CSV
            # -------------------------------
            csv = sorted_perf_individual.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Summary as CSV",
                data=csv,
                file_name="asset_performance_summary_individual_assets.csv",
                mime="text/csv",
            )
