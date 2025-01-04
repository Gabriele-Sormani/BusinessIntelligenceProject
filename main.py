import streamlit as st
import pandas as pd
from streamlit_extras.stylable_container import stylable_container
import altair as alt

st.set_page_config(
    page_title="Financial News Sentiment Analysis",
    layout="wide")

df = pd.read_csv('query_results.csv')
queries = df['query'].unique()

st.title("Financial News Sentiment Analysis")

st.subheader("Query overview")
with stylable_container(
        key="elevated_container",
        css_styles="""
            {
                background-color: rgba(95, 100, 140, 0.1);  
                border: 1px solid rgba(95, 100, 140, 0.4);  
                border-radius: 10px;  
                padding: 20px;  
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  
            }
            
            
        """
):
    with st.container():
        option = st.selectbox(
            "Select a query",
            queries,
        )

        query_data = df[df['query'] == option]
        query_data = query_data.rename(columns={"sentiment": "predicted sentiment", "gt": "ground truth sentiment"})

        event = st.dataframe(
            query_data[['text', 'distance', 'predicted sentiment', 'ground truth sentiment']],
            on_select='rerun',
            selection_mode='single-row',
            hide_index=True,
            use_container_width=True,
            column_config={
                "text": st.column_config.TextColumn(),
                "distance": st.column_config.Column(width='small'),
                "predicted sentiment": st.column_config.Column(width='small'),
                "ground truth sentiment": st.column_config.Column(width='small')
            },
        )

        if len(event.selection['rows']):
            selected_row = event.selection['rows'][0]
            full_text = (query_data.iloc[selected_row]['text'])
            st.text('Full text:')
            with st.container(border=True):
                st.text(full_text)

st.subheader("Explainability")
with stylable_container(
        key="elevated_container",
        css_styles="""
            {
                background-color: rgba(95, 100, 140, 0.1);  
                border: 1px solid rgba(95, 100, 140, 0.4);  
                border-radius: 10px;  
                padding: 20px;  
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);  
            }


        """
):
    with st.container():
        if len(event.selection['rows']):
            selected_row = event.selection['rows'][0]
            index = (int(query_data.iloc[selected_row].name))
            barplot_name = f'data_barplot{index}.csv'
            interpret_name = f'transformer_interpret{index}.html'
            barplot_data = pd.read_csv(f"./results_images/{barplot_name}")

            st.subheader('Attention barplot')
            bars = (
                alt.Chart(barplot_data)
                .mark_bar()
                .encode(
                    alt.X('mean(value)'),
                    alt.Y('tokens', sort=alt.EncodingSortField(field="value", op="mean", order='descending'))
                ).transform_window(
                    rank='rank(value)',
                    sort=[alt.SortField('value', order='descending')]
                ).transform_filter(
                    (alt.datum.rank < 15))
            )

            st.altair_chart(bars, theme="streamlit", use_container_width=True)

            st.subheader('Transformer explainer')
            st.html(f"./results_images/{interpret_name}")
        else:
            st.text('Choose a specific result to see the explanation')
