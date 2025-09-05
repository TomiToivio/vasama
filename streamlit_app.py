import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from streamlit_agraph import agraph, Node, Edge, Config
from datetime import datetime, timedelta

def _count_series_to_df(s: pd.Series, name="count", index_name="value"):
    df = s.reset_index()
    df.columns = [index_name, name]
    return df

def bar_counts(series: pd.Series, title=None, x_label=None, y_label="Count", orientation="v"):
    dfc = _count_series_to_df(series, name="count", index_name="label")
    fig = px.bar(
        dfc, x="label" if orientation=="v" else "count",
        y="count" if orientation=="v" else "label",
        orientation=orientation, title=title
    )
    fig.update_layout(
        xaxis_title=x_label, yaxis_title=y_label,
        margin=dict(t=60, r=20, b=40, l=60), legend_title=None
    )
    fig.update_traces(hovertemplate="%{x}<br>Count=%{y}" if orientation=="v" else "%{y}<br>Count=%{x}")
    return fig

def stacked_daily(df, date_col, category_col, title):
    # expects df[date_col] already datetime
    daily = df.groupby([date_col, category_col]).size().reset_index(name="count")
    fig = px.area(  # nice for multi-class over time; use px.bar with barmode="stack" if you prefer
        daily, x=date_col, y="count", color=category_col, title=title,
        groupnorm=None  # set to 'fraction' if you want % share
    )
    fig.update_layout(xaxis_title="Date", yaxis_title="Count", hovermode="x unified", margin=dict(t=60, r=20, b=40, l=60))
    return fig

def stacked_bar_by(df, by_col, category_col, title):
    counts = df.groupby([by_col, category_col]).size().reset_index(name="count")
    fig = px.bar(counts, x=by_col, y="count", color=category_col, title=title, barmode="stack")
    fig.update_layout(xaxis_title=by_col.capitalize(), yaxis_title="Count", margin=dict(t=60, r=20, b=40, l=60))
    return fig

def table_topic_words(topic_col, words_col, df, title):
    # one row per topic (first words seen)
    temp = df[[topic_col, words_col]].dropna().astype(str)
    first_words = temp.groupby(topic_col)[words_col].first().reset_index()
    counts = df[topic_col].astype(str).value_counts().reset_index()
    counts.columns = [topic_col, "Count"]
    merged = counts.merge(first_words, on=topic_col, how="left")
    merged = merged.sort_values("Count", ascending=False)

    fig = go.Figure(
        data=[go.Table(
            header=dict(values=[topic_col, "Count", words_col]),
            cells=dict(values=[merged[topic_col], merged["Count"], merged[words_col]])
        )]
    )
    fig.update_layout(title=title, margin=dict(t=60, r=20, b=40, l=20))
    return fig

def osint_map(map_coordinates):
    map_coordinates = eval(map_coordinates)
    # get last lat and lng
    last_coord = map_coordinates[-1]
    lat = last_coord.get("lat", 48.8891738)
    lng = last_coord.get("lng", 30.922972)
    # 48.8891738,30.922972,6.5z
    geomap = folium.Map(location=[lat, lng], zoom_start=16)
    for coord in map_coordinates:
        event = coord.get("event", "N/A")
        lat = coord.get("lat")
        lng = coord.get("lng")
        location = coord.get("location", "N/A")
        description = coord.get("description", "N/A")
        popup_text = f"Event: {event}<br>Description: {description}"
        # Change icon
        folium.Marker(
            [lat, lng], popup=popup_text, tooltip=location, icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(geomap)
    # call to render Folium map in Streamlit
    st_data = st_folium(geomap, width=725)

def osint_graph(edges_list):
    nodes = []
    edges = []
    for edge in edges_list:
        source = edge["source"]
        target = edge["target"]
        description = edge["description"]
        if source and target:
            # check if node already in nodes
            if not any(node.id == source for node in nodes):
                nodes.append(Node(id=source))
            if not any(node.id == target for node in nodes):
                nodes.append(Node(id=target))
            # check if edge already in edges
            edges.append(Edge(source=source, target=target, label=description))
    config = Config(width="100%", height=400, directed=True, physics=True, hierarchical=False)
    return agraph(nodes=nodes, edges=edges, config=config)


def osint_map_multiple(all_map_coordinates):
    cleaned_coordinates = []
    # 48.8891738,30.922972,6.5z
    geomap = folium.Map(location=[48.8891738, 30.922972], zoom_start=7)
    for map_coordinate_list in all_map_coordinates:
        map_coordinate_list = eval(map_coordinate_list)
        # If not empty, add to cleaned_coordinates
        if map_coordinate_list:
            # if has lat and lng
            for coord in map_coordinate_list:
                if "lat" in coord and "lng" in coord:
                    cleaned_coordinates.append(coord)
    for coordinates in cleaned_coordinates:
        event = coordinates.get("event", "N/A")
        lat = coordinates.get("lat")
        lng = coordinates.get("lng")
        location = coordinates.get("location", "N/A")
        description = coordinates.get("description", "N/A")
        popup_text = f"Event: {event}<br>Description: {description}"
        # Change icon
        folium.Marker(
            [lat, lng], popup=popup_text, tooltip=location, icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(geomap)
    st_data = st_folium(geomap, width="100%")

def dataframe_with_selections(df: pd.DataFrame, init_value: bool = False) -> pd.DataFrame:
    global selected_index
    previous_selection = []
    column_config = {
        "Select": st.column_config.CheckboxColumn(
            label="Display",
            help="Select the message to display.",
            default=init_value,
            required=True,
        ),  
        "message_url": st.column_config.TextColumn(
            label="URL",
            help="The URL for the message.",
        ),
        "message_date": st.column_config.DateColumn(
            label="Date",
            help="The date when the message was posted.",
        ),
        "channel_url": st.column_config.TextColumn(
            label="Channel URL",
            help="The URL of the channel.",
        ),
        "message_text": st.column_config.TextColumn(
            label="Text",
            help="Message text.",
        ),
        "translated_text": st.column_config.TextColumn(
            label="Translated Text",
            help="Translated message text.",
        ),
        "message_language": st.column_config.TextColumn(
            label="Language",
            help="The language of the message.",
        ),
        "multimodal_analysis": st.column_config.TextColumn(
            label="Multimodal Analysis",
            help="Analysis of the multimodal content of the message.",
        ),
        "political_analysis": st.column_config.TextColumn(
            label="Political Analysis",
            help="Analysis of the political content of the message.",
        ),
        "osint_analysis": st.column_config.TextColumn(
            label="OSINT Analysis",
            help="Analysis of the OSINT content of the message.",
        ),
        "osint_entities": st.column_config.TextColumn(
            label="OSINT Entities",
            help="Entities related to OSINT mentioned in the message.",
        ),
        "osint_topics": st.column_config.TextColumn(
            label="OSINT Topics",
            help="Topics related to OSINT mentioned in the message.",
        ),
        "negative_sentiments": st.column_config.TextColumn(
            label="Negative Sentiments",
            help="Negative sentiments detected in the message.",
        ),
        "neutral_sentiments": st.column_config.TextColumn(
            label="Neutral Sentiments",
            help="Neutral sentiments detected in the message.",
        ),
        "positive_sentiments": st.column_config.TextColumn(
            label="Positive Sentiments",
            help="Positive sentiments detected in the message.",
        ),
        "whisper_language": st.column_config.TextColumn(
            label="Whisper Language",
            help="Language detected in the whisper transcript.",
        ),
        "whisper_transcript": st.column_config.TextColumn(
            label="Whisper Transcript",
            help="Transcript of the whisper audio.",
        ),
        "whisper_translated": st.column_config.TextColumn(
            label="Whisper Translated",
            help="Translated text of the whisper audio.",
        ),
        "map_coordinates": st.column_config.TextColumn(
            label="Map Coordinates",
            help="Geographical coordinates mentioned in the message.",
        ),
        "network_edges": st.column_config.TextColumn(
            label="Network Edges",
            help="Network edges mentioned in the message.",
        ),
        "timeline_dates": st.column_config.TextColumn(
            label="Timeline Dates",
            help="Timeline dates mentioned in the message.",
        ),
        "channel_id": None,
        "geo_locations": None,
        "message_id": None,
        "message_raw_text": None,
        "ocr_text": None,
        "russian_entities": None,
        "spacy_entities": None,
        "ukrainian_entities": None,
    }
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", init_value)
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config=column_config,
        column_order=["Select","message_url","message_date","message_text","translated_text","message_language","multimodal_analysis","osint_analysis","political_analysis","osint_topics","osint_entities","positive_sentiments","neutral_sentiments","negative_sentiments","whisper_language","whisper_transcript","whisper_translated","map_coordinates","network_edges","timeline_dates"],
        disabled=df.columns,
    )
    selected_rows = edited_df[edited_df.Select]
    return selected_rows.drop('Select', axis=1)

def vasama_dashboard():
    global df, selected_index

    if "selected_row_index" not in st.session_state:
        st.session_state.selected_row_index = None
    else:   
        selected_index = st.session_state.selected_row_index

    search_query = st.sidebar.text_input("Search dataframe:")
    osint_topics = df["osint_topics"].dropna().str.split(",").explode().str.strip().unique().tolist()
    # Order alphabetically
    osint_topics.sort()
    osint_topics.insert(0, "All")
    osint_topic_filter_selection = st.sidebar.selectbox(
        "Select OSINT topic:",
        options=osint_topics,
        index=0,
        help="Filter by topics."
    )
    osint_entities = df["osint_entities"].dropna().str.split(",").explode().str.strip().unique().tolist()
    # Order alphabetically
    osint_entities.sort()
    osint_entities.insert(0, "All")
    osint_entity_filter_selection = st.sidebar.selectbox(
        "Select OSINT entity:",
        options=osint_entities,
        index=0,
        help="Filter by entities."
    )
    positive_sentiments = df["positive_sentiments"].dropna().str.split(",").explode().str.strip().unique().tolist()
    # Order alphabetically
    positive_sentiments.sort()
    positive_sentiments.insert(0, "All")
    positive_sentiment_filter_selection = st.sidebar.selectbox(
        "Select positive sentiment:",
        options=positive_sentiments,
        index=0,  
        help="Filter by positive sentiments."
    )
    neutral_sentiments = df["neutral_sentiments"].dropna().str.split(",").explode().str.strip().unique().tolist()
    # Order alphabetically
    neutral_sentiments.sort()
    neutral_sentiments.insert(0, "All")
    neutral_sentiment_filter_selection = st.sidebar.selectbox(
        "Select neutral sentiment:",
        options=neutral_sentiments,
        index=0,  
        help="Filter by neutral sentiments."
    )
    negative_sentiments = df["negative_sentiments"].dropna().str.split(",").explode().str.strip().unique().tolist()
    # Order alphabetically
    negative_sentiments.sort()
    negative_sentiments.insert(0, "All")
    negative_sentiment_filter_selection = st.sidebar.selectbox(
        "Select negative sentiment:",
        options=negative_sentiments,
        index=0,  
        help="Filter by negative sentiments."
    )
    # Add start and end date filter
    min_date = df["message_date"].min().date()
    max_date = df["message_date"].max().date()
    # Add default end date 2025-08-24
    default_end_date = datetime(2025, 8, 24).date()
    # add default start date 2025-08-23
    default_start_date = datetime(2025, 8, 23).date()
    # Add start date filter
    start_date = st.sidebar.date_input(
        "Select start date:",
        value=default_start_date,
        min_value=min_date,
        max_value=max_date,
        help="Filter by start date."
    )
    # Add end date filter
    end_date = st.sidebar.date_input(
        "Select end date:",
        value=default_end_date,
        min_value=min_date,
        max_value=max_date,
        help="Filter by end date."
    )
    if start_date > end_date:
        st.sidebar.error("Error: End date must fall after start date.")
    df = df[(df["message_date"].dt.date >= start_date) & (df["message_date"].dt.date <= end_date)]
    filtered_df = df.copy()
    if osint_topic_filter_selection == "All":
        osint_topic_filter = df["osint_topics"].unique().tolist()
    else:
        osint_topic_filter = [osint_topic_filter_selection]
    filtered_df = filtered_df[(filtered_df["osint_topics"].isin(osint_topic_filter))]
    if osint_entity_filter_selection == "All":
        osint_entity_filter = df["osint_entities"].unique().tolist()
    else:
        osint_entity_filter = [osint_entity_filter_selection]
    filtered_df = filtered_df[(filtered_df["osint_entities"].isin(osint_entity_filter))]
    if positive_sentiment_filter_selection == "All":
        positive_sentiment_filter = df["positive_sentiments"].unique().tolist()
    else:
        positive_sentiment_filter = [positive_sentiment_filter_selection]
    filtered_df = filtered_df[(filtered_df["positive_sentiments"].isin(positive_sentiment_filter))]
    if neutral_sentiment_filter_selection == "All":
        neutral_sentiment_filter = df["neutral_sentiments"].unique().tolist()
    else:
        neutral_sentiment_filter = [neutral_sentiment_filter_selection]
    filtered_df = filtered_df[(filtered_df["neutral_sentiments"].isin(neutral_sentiment_filter))]
    if negative_sentiment_filter_selection == "All":
        negative_sentiment_filter = df["negative_sentiments"].unique().tolist()
    else:
        negative_sentiment_filter = [negative_sentiment_filter_selection]
    filtered_df = filtered_df[(filtered_df["negative_sentiments"].isin(negative_sentiment_filter))]
    if negative_sentiment_filter_selection == "All":
        negative_sentiment_filter = df["negative_sentiments"].unique().tolist()
    else:
        negative_sentiment_filter = [negative_sentiment_filter_selection]
    filtered_df = filtered_df[(filtered_df["negative_sentiments"].isin(negative_sentiment_filter))]
    if search_query:
        filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)]
    selection = dataframe_with_selections(filtered_df)

    if not selection.empty:
        for index, row in selection.iterrows():
            message_data = row.to_dict()
            message_url = message_data.get("message_url", "N/A")
            message_text = message_data.get("message_text", "N/A")
            message_date = message_data.get("message_date", "N/A")
            message_username = message_data.get("message_username", "N/A")
            message_translated = message_data.get("translated_text", "N/A")
            multimodal_analysis = message_data.get("multimodal_analysis", "N/A")
            osint_analysis = message_data.get("osint_analysis", "N/A")
            osint_entities = message_data.get("osint_entities", "N/A")
            osint_topics = message_data.get("osint_topics", "N/A")
            political_analysis = message_data.get("political_analysis", "N/A")
            negative_sentiments = message_data.get("negative_sentiments", "N/A")
            neutral_sentiments = message_data.get("neutral_sentiments", "N/A")
            positive_sentiments = message_data.get("positive_sentiments", "N/A")
            political_themes = message_data.get("political_themes", "N/A")
            whisper_transcript = message_data.get("whisper_transcript", "N/A")
            whisper_language = message_data.get("whisper_language", "N/A")
            whisper_translated = message_data.get("whisper_translated", "N/A")
            map_coordinates = message_data.get("map_coordinates", "N/A")
            network_edges = message_data.get("network_edges", "N/A")
            col1, col2 = st.columns([0.3, 0.7])
            with col1:
                st.subheader("Message Details")
                st.write("**Message URL:**", message_url)
                st.write("**Message Date:**", message_date)
                st.subheader("Message Text")
                message_text = message_text.replace("\\n", "<br>")
                st.markdown(message_text, unsafe_allow_html=True)
                st.subheader("Translated Text")
                message_translated = message_translated.replace("\\n", "<br>")
                st.markdown(message_translated, unsafe_allow_html=True)
            with col2:
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Multimodal", "OSINT", "Political", "Entities", "Sentiments", "Topics", "Whisper", "Map"])
                with tab1:
                    #st.subheader("Multimodal Analysis")
                    # Convert "\n"
                    st.header("Multimodal Analysis")
                    multimodal_analysis = multimodal_analysis.replace('\\n', '<br>')
                    st.markdown(multimodal_analysis, unsafe_allow_html=True)
                with tab2:
                    st.subheader("OSINT Analysis")
                    osint_analysis = osint_analysis.replace("\\n", "<br>")
                    st.markdown(osint_analysis, unsafe_allow_html=True)
                with tab3:
                    st.subheader("Political Analysis")
                    political_analysis = political_analysis.replace("\\n", "<br>")
                    st.markdown(political_analysis, unsafe_allow_html=True)
                with tab4:
                    st.subheader("Entities")
                    st.write(str(osint_entities))
                with tab5:
                    st.subheader("Sentiments")
                    st.write("**Positive Sentiments:**", str(positive_sentiments))
                    st.write("**Neutral Sentiments:**", str(neutral_sentiments))
                    st.write("**Negative Sentiments:**", str(negative_sentiments))
                with tab6:
                    st.subheader("Topics")
                    st.write("**OSINT Topics:**", str(osint_topics))
                with tab7:
                    st.subheader("Whisper")
                    st.write("**Whisper Transcript:**", str(whisper_transcript))
                    st.write("**Whisper Language:**", str(whisper_language))
                    st.write("**Whisper Translated:**", str(whisper_translated))
                with tab8:
                    st.subheader("Coordinates")
                    # If map coordinates is not empty
                    if map_coordinates and map_coordinates != "N/A" and map_coordinates != "[]":
                        osint_map(map_coordinates)
                    else:
                        st.write("No map coordinates available for this message.")
                with tab8:
                    st.subheader("Graph")
                    # If network edges is not empty
                    if network_edges and network_edges != "N/A" and network_edges != "[]":
                        osint_graph([network_edges])
                    else:
                        st.write("No network edges available for this message.")


    if selection.empty:
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Sentiments", "Entities", "Topics", "Graph", "Map"])

        with tab1:
            st.subheader("Number of Messages")
            total_message_count = len(df)
            filtered_message_count = len(filtered_df)
            st.write(f"Total messages: {total_message_count}")
            st.write(f"Filtered messages: {filtered_message_count}")
            # Messages by day
            st.subheader("Top Positive Sentiments")
            pos = filtered_df["positive_sentiments"].dropna().astype(str).str.split(", ").explode().value_counts().head(20)
            st.plotly_chart(bar_counts(pos, title="Top positive", x_label="Positive"), use_container_width=True)

            st.subheader("Top Neutral Sentiments")
            neu = filtered_df["neutral_sentiments"].dropna().astype(str).str.split(", ").explode().value_counts().head(20)
            st.plotly_chart(bar_counts(neu, title="Top neutral", x_label="Neutral"), use_container_width=True)

            st.subheader("Top Negative Sentiments")
            neg = filtered_df["negative_sentiments"].dropna().astype(str).str.split(", ").explode().value_counts().head(20)
            st.plotly_chart(bar_counts(neg, title="Top negative", x_label="Negative"), use_container_width=True)

        with tab2:
            st.subheader("Top Entities")
            entities_series = filtered_df["osint_entities"].dropna().astype(str).str.split(", ").explode()
            top_entities = entities_series.value_counts().head(20)
            # horizontal bar is often easier to read for long labels put ones with largest count to top
            fig = bar_counts(top_entities, title="Top 20 entities", x_label="Count", orientation="v")
            st.plotly_chart(fig, use_container_width=True)
        with tab3:
            # Most Frequent Topics
            st.subheader("Top Topics")
            topics_series = filtered_df["osint_topics"].dropna().astype(str).str.split(", ").explode()
            top_topics = topics_series.value_counts().head(20)
            fig = bar_counts(top_topics, title="Top 20 topics", x_label="Topic")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("Network Graph")
            all_edges = filtered_df["network_edges"].dropna().astype(str)
            new_edges_list = []
            for edges in all_edges:
                # IF not empty
                if edges and edges != "N/A" and edges != "[]":
                    edges = eval(edges)
                    new_edges_list.extend(edges)
            osint_graph(new_edges_list)

        with tab5:
            st.subheader("Event Map")
            all_map_coordinates = filtered_df["map_coordinates"].dropna().astype(str)
            # remove empty coordinates
            osint_map_multiple(all_map_coordinates)




st.set_page_config(
    page_title="Vasama Telegram Data Analysis Demo",
    page_icon=":alien:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/TomiToivio/vasama/issues',
        'Report a bug': "https://github.com/TomiToivio/vasama/issues",
        'About': f"""This is a demo dashboard of the Vasama data collection and analysis pipeline.
        Data is collected from [Telegram](https://web.telegram.org/) and analyzed with [Ollama](https://ollama.com/).
        Based on data analysis pipeline developed by [Tomi Toivio](mailto:tomi.toivio@helsinki.fi).
        Vasama [GitHub Repository](https://github.com/TomiToivio/vasama).
        """
    }
)

@st.cache_data
def load_data():
    df = pd.read_csv("vasama_dataframe.csv", engine='python', encoding='utf-8')
    # Convert message_date to datetime
    df["message_date"] = pd.to_datetime(df["message_date"], errors='coerce')
    return df

df = load_data()


# On rerun
if "selected_row_index" not in st.session_state:
    st.session_state.selected_row_index = None
else:   
    selected_index = st.session_state.selected_row_index

st.title("Vasama: Telegram Data Analysis Demo")

vasama_dashboard()

st.caption("Vasama: Telegram Data Analysis Demo")
