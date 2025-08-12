import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def dataframe_with_selections(df: pd.DataFrame, init_value: bool = False) -> pd.DataFrame:
    global selected_index
    previous_selection = []
    column_config = {
        "Select": st.column_config.CheckboxColumn(
            label="Display Status",
            help="Select the status to display.",
            default=init_value,
            required=True,
        ),  
        "status_id": st.column_config.TextColumn(
            label="Status ID",
            help="The unique identifier for the post.",
        ),
        "status_date": st.column_config.DateColumn(
            label="Status Date",
            help="The date when the status was posted.",
        ),
        "status_url": st.column_config.TextColumn(
            label="Status URL",
            help="The URL of the status.",
        ),
        "status_text": st.column_config.TextColumn(
            label="Text",
            help="Status text.",
        ),
        "status_translated": st.column_config.TextColumn(
            label="Translated Text",
            help="Translated status text.",
        ),
        "status_username": st.column_config.TextColumn(
            label="Username",
            help="The username of the account that posted the status.",
        ),
        "political_analysis": st.column_config.TextColumn(
            label="Political Analysis",
            help="Analysis of the political content of the status.",
        ),
        "political_entities": st.column_config.TextColumn(
            label="Political Entities",
            help="Entities related to politics mentioned in the status.",
        ),
        "political_topics": st.column_config.TextColumn(
            label="Political Topics",
            help="Topics related to politics mentioned in the status.",
        ),
        "negative_sentiments": st.column_config.TextColumn(
            label="Negative Sentiments",
            help="Negative sentiments detected in the status.",
        ),
        "neutral_sentiments": st.column_config.TextColumn(
            label="Neutral Sentiments",
            help="Neutral sentiments detected in the status.",
        ),
        "positive_sentiments": st.column_config.TextColumn(
            label="Positive Sentiments",
            help="Positive sentiments detected in the status.",
        ),
        "political_themes": st.column_config.TextColumn(
            label="Political Themes",
            help="Themes related to politics detected in the status.",
        ),
        "status_source": None,
        "status_processed": None,
    }
    df_with_selections = df.copy()
    df_with_selections.insert(0, "Select", init_value)
    edited_df = st.data_editor(
        df_with_selections,
        hide_index=True,
        column_config=column_config,
        column_order=["Select","status_id","status_date","status_url","status_text","status_translated","status_username","political_analysis","political_entities","political_topics","negative_sentiments","neutral_sentiments","positive_sentiments","political_themes"],
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
    political_themes = df["political_themes"].dropna().str.split(",").explode().str.strip().unique().tolist()
    political_themes.insert(0, "All")
    political_theme_filter_selection = st.sidebar.selectbox(
        "Select political theme:",
        options=political_themes,
        index=0,  
        help="Filter by political themes."
    )
    political_topics = df["political_topics"].dropna().str.split(",").explode().str.strip().unique().tolist()
    political_topics.insert(0, "All")
    political_topic_filter_selection = st.sidebar.selectbox(
        "Select political topic:",
        options=political_topics,
        index=0,  
        help="Filter by political topics."
    )
    political_entities = df["political_entities"].dropna().str.split(",").explode().str.strip().unique().tolist()
    political_entities.insert(0, "All")
    political_entity_filter_selection = st.sidebar.selectbox(
        "Select political entity:",
        options=political_entities,
        index=0,  
        help="Filter by political entities."
    )
    positive_sentiments = df["positive_sentiments"].dropna().str.split(",").explode().str.strip().unique().tolist()
    positive_sentiments.insert(0, "All")
    positive_sentiment_filter_selection = st.sidebar.selectbox(
        "Select positive sentiment:",
        options=positive_sentiments,
        index=0,  
        help="Filter by positive sentiments."
    )
    neutral_sentiments = df["neutral_sentiments"].dropna().str.split(",").explode().str.strip().unique().tolist()
    neutral_sentiments.insert(0, "All")
    neutral_sentiment_filter_selection = st.sidebar.selectbox(
        "Select neutral sentiment:",
        options=neutral_sentiments,
        index=0,  
        help="Filter by neutral sentiments."
    )
    negative_sentiments = df["negative_sentiments"].dropna().str.split(",").explode().str.strip().unique().tolist()
    negative_sentiments.insert(0, "All")
    negative_sentiment_filter_selection = st.sidebar.selectbox(
        "Select negative sentiment:",
        options=negative_sentiments,
        index=0,  
        help="Filter by negative sentiments."
    )
    # 
    filtered_df = df.copy()
    if political_theme_filter_selection == "All":
        political_theme_filter = df["political_themes"].unique().tolist()
    else:
        political_theme_filter = [political_theme_filter_selection]
    filtered_df = filtered_df[(filtered_df["political_themes"].isin(political_theme_filter))]
    if political_topic_filter_selection == "All":
        political_topic_filter = df["political_topics"].unique().tolist()
    else:
        political_topic_filter = [political_topic_filter_selection]
    filtered_df = filtered_df[(filtered_df["political_topics"].isin(political_topic_filter))]
    if political_entity_filter_selection == "All":
        political_entity_filter = df["political_entities"].unique().tolist()
    else:
        political_entity_filter = [political_entity_filter_selection]
    filtered_df = filtered_df[(filtered_df["political_entities"].isin(political_entity_filter))]
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
    if search_query:
        filtered_df = filtered_df[filtered_df.apply(lambda row: row.astype(str).str.contains(search_query, case=False, na=False).any(), axis=1)]
    selection = dataframe_with_selections(filtered_df)

    if not selection.empty:
        for index, row in selection.iterrows():
            status_data = row.to_dict()
            status_id = status_data.get("status_id", "N/A")
            status_url = status_data.get("status_url", "N/A")
            status_text = status_data.get("status_text", "N/A")
            status_date = status_data.get("status_date", "N/A")
            status_username = status_data.get("status_username", "N/A")
            status_translated = status_data.get("status_translated", "N/A")
            political_analysis = status_data.get("political_analysis", "N/A")
            political_entities = status_data.get("political_entities", "N/A")
            political_topics = status_data.get("political_topics", "N/A")
            negative_sentiments = status_data.get("negative_sentiments", "N/A")
            neutral_sentiments = status_data.get("neutral_sentiments", "N/A")
            positive_sentiments = status_data.get("positive_sentiments", "N/A")
            political_themes = status_data.get("political_themes", "N/A")
            col1, col2 = st.columns([0.3, 0.7])
            with col1:
                st.subheader("Status Details")
                st.write("**Status ID:**", status_id)
                st.write("**Status Date:**", status_date)
                st.write("**Username:**", status_username)
                st.write("**Status URL:**", status_url)
                st.write("**Status Text:**", status_text)
                st.write("**Translated Text:**", status_translated)
            with col2:
                tab1, tab2, tab3, tab4, tab5 = st.tabs(["Analysis", "Entities", "Sentiments", "Topics", "Themes"])
                with tab1:
                    st.subheader("Analysis")
                    st.write("**Political Analysis:**", political_analysis)
                with tab2:
                    st.subheader("Entities")
                    st.write("**Political Entities:**", political_entities)
                with tab3:
                    st.subheader("Sentiments")
                    st.write("**Positive Sentiments:**", positive_sentiments)
                    st.write("**Neutral Sentiments:**", neutral_sentiments)
                    st.write("**Negative Sentiments:**", negative_sentiments)
                with tab4:
                    st.subheader("Topics")
                    st.write("**Political Topics:**", political_topics)
                with tab5:
                    st.subheader("Themes")
                    st.write("**Political Themes:**", political_themes)


    if selection.empty:
        tab1, tab2, tab3, tab4 = st.tabs(["Frequency", "Entities", "Topics", "Sentiments"])
        with tab1:
            st.subheader("Number of Statuses")
            total_status_count = len(df)
            filtered_status_count = len(filtered_df)
            st.write(f"Total statuses: {total_status_count}")
            st.write(f"Filtered statuses: {filtered_status_count}")
        with tab2:
            st.subheader("Top Entities")
            entities_series = filtered_df["political_entities"].dropna().str.split(", ").explode()
            top_entities = entities_series.value_counts().head(20)
            top_entities = top_entities.sort_values(ascending=False)
            st.bar_chart(top_entities)
        with tab3:
            st.subheader("Top Topics")
            topics_series = filtered_df["political_topics"].dropna().str.split(", ").explode()
            top_topics = topics_series.value_counts().head(20)
            top_topics = top_topics.sort_values(ascending=False)
            st.bar_chart(top_topics)
        with tab4:

            st.subheader("Top Positive Sentiments")
            positive_series = filtered_df["positive_sentiments"].dropna().str.split(", ").explode()
            top_positive = positive_series.value_counts().head(20)
            top_positive = top_positive.sort_values(ascending=False)
            st.bar_chart(top_positive)

            st.subheader("Top Neutral Sentiments")
            neutral_series = filtered_df["neutral_sentiments"].dropna().str.split(", ").explode()
            top_neutral = neutral_series.value_counts().head(20)
            top_neutral = top_neutral.sort_values(ascending=False)
            st.bar_chart(top_neutral)

            st.subheader("Top Negative Sentiments")
            negative_series = filtered_df["negative_sentiments"].dropna().str.split(", ").explode()
            top_negative = negative_series.value_counts().head(20)
            top_negative = top_negative.sort_values(ascending=False)
            st.bar_chart(top_negative)

st.set_page_config(
    page_title="Vasama Mastodon Data Analysis Demo",
    page_icon=":alien:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://mastodon.social/@toivio',
        'Report a bug': "https://github.com/TomiToivio/vasama/issues",
        'About': f"""This is a demo dashboard of the Vasama data collection and analysis pipeline.
        Data is collected from [Mastodon](https://mastodon.social/) and analyzed with [Ollama](https://ollama.com/).
        Based on data analysis pipeline developed by [Tomi Toivio](mailto:tomi.toivio@helsinki.fi) in the University of Helsinki.
        Vasama [GitHub Repository](https://github.com/TomiToivio/vasama).
        """
    }
)

@st.cache_data
def load_data():
    return pd.read_csv("statuses.csv", engine='python')

df = load_data()


# On rerun
if "selected_row_index" not in st.session_state:
    st.session_state.selected_row_index = None
else:   
    selected_index = st.session_state.selected_row_index

st.title("Vasama: Mastodon Data Analysis Demo")

vasama_dashboard()

st.caption("Vasama: Mastodon Data Analysis Demo")
