import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO
import chardet
import openpyxl
from PIL import Image
import requests
from io import BytesIO
import re

st.title("Responsive Display Asset Analysis")

# Allow multiple file uploads
files = st.file_uploader("Upload CSV files", type="csv", accept_multiple_files=True)

def detect_encoding(file):
    """Detects the encoding of the file content."""
    raw_data = file.read()
    result = chardet.detect(raw_data)
    encoding = result['encoding']
    file.seek(0)  # Reset file pointer after reading
    return encoding

if files:
    if st.button("Process"):
        combined_data = []
        line1 = None
        line2 = None
        header_line = None

        for idx, file in enumerate(files):
            # Detect file encoding
            encoding = detect_encoding(file)

            try:
                # Read the file contents with the detected encoding
                content = file.getvalue().decode(encoding)
            except UnicodeDecodeError:
                st.error(f"Could not decode the file {file.name}. Please check the encoding.")
                continue

            lines = content.splitlines()

            # For the first file, keep track of the first two lines (header lines) and the actual header
            if idx == 0:
                line1 = lines[0]  # Report title
                line2 = lines[1]  # Date range
                header_line = lines[2].strip().split('\t')  # Split the headers correctly by tab
                data_lines = lines[3:]  # Remaining data
            else:
                data_lines = lines[3:]  # Skip the header lines for subsequent files, including the actual header

            # Convert data lines into DataFrame
            data = [line.split('\t') for line in data_lines if line.strip()]  # Split by tab and ignore empty lines
            df = pd.DataFrame(data, columns=header_line)

            # Add the filename column
            df['Filename'] = file.name

            combined_data.append(df)

        # Concatenate all DataFrames
        combined_df = pd.concat(combined_data, ignore_index=True)

        # Standardize the header names to lowercase for consistency
        combined_df.columns = combined_df.columns.str.lower()

        # Filter out assets with a status of 'Removed'
        combined_df = combined_df[combined_df['asset status'].str.lower() != 'removed']

        # Check if required columns exist
        required_columns = ['asset status', 'asset', 'status', 'asset type', 'performance']
        if all(col in combined_df.columns for col in required_columns):
            # Aggregate the data to get unique assets with summed performance metrics
            combined_df['performance'] = combined_df['performance'].str.title()  # Standardize performance values for consistency
            performance_summary = combined_df.pivot_table(index=['asset', 'asset type'], columns='performance', aggfunc='size', fill_value=0).reset_index()
            performance_summary.columns.name = None  # Remove the columns name for cleaner display

            # Ensure all expected performance columns are present
            selected_columns = ['Best', 'Good', 'Low']
            for col in selected_columns:
                if col not in performance_summary.columns:
                    performance_summary[col] = 0

            # Reorder columns for display
            display_columns = ['asset', 'asset type'] + selected_columns
            performance_summary = performance_summary.reindex(columns=display_columns, fill_value=0)

            # Separate data by asset type for display
            st.subheader("Overall Asset Performance Summary by Asset Type (Best, Good, Low)")
            asset_types = performance_summary['asset type'].unique()
            for asset_type in asset_types:
                filtered_df = performance_summary[performance_summary['asset type'] == asset_type]
                if not filtered_df.empty:
                    st.markdown(f"### {asset_type.title()}")
                    for index, row in filtered_df.iterrows():
                        asset_url = row['asset']
                        asset_type = row['asset type'].lower()

                        # Create columns for each row in the table
                        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                        with col1:
                            # Display asset preview directly inside the table cell with smaller thumbnails
                            if asset_type in ['square image', 'landscape image', 'logo', 'landscape logo']:
                                if re.match(r'^https?://', asset_url):
                                    try:
                                        response = requests.get(asset_url)
                                        if response.status_code == 200:
                                            img = Image.open(BytesIO(response.content))
                                            st.image(img, caption=asset_url.split('/')[-1], use_column_width=False, width=100)
                                        else:
                                            st.write("Invalid Image URL")
                                    except Exception as e:
                                        st.write(f"Error loading image: {e}")
                            elif asset_type == 'youtube video':
                                youtube_match = re.search(r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w-]+)', asset_url)
                                if youtube_match:
                                    video_id = youtube_match.group(1)
                                    st.video(f"https://www.youtube.com/embed/{video_id}")
                                else:
                                    st.write("Invalid YouTube URL")
                            else:
                                st.markdown(f"<div style=\"word-wrap: break-word;\">{asset_url.split('/')[-1]}</div>", unsafe_allow_html=True)

                        # Display performance metrics in the remaining columns
                        with col2:
                            st.write(f"Best: {row['Best']}")
                        with col3:
                            st.write(f"Good: {row['Good']}")
                        with col4:
                            st.write(f"Low: {row['Low']}")

                        # Justify the content in all rows to be vertically centered
                        st.markdown(
                            """
                            <style>
                            [data-testid="column"] { display: flex; align-items: center; }
                            </style>
                            """,
                            unsafe_allow_html=True
                        )

        else:
            st.error("The uploaded files do not contain the required columns: 'Asset status', 'Asset', 'Status', 'Asset type', 'Performance'.")