import pprint

import pandas as pd
import streamlit as st
from transformers import pipeline
import plotly.express as px
from pipe import select

# import streamlit_utilities as su


model_path = "SamLowe/roberta-base-go_emotions"
classifier = pipeline("text-classification",
                     model=model_path,
                     tokenizer=model_path,
                     max_length=512, truncation=True, top_k=None)


EMOTIONS_POSITIVE = {'neutral', 'realization', 'approval', 'admiration', 'amusement', 'surprise', 'curiosity', 'optimism', 'joy', 'desire', 'excitement', 'caring', 'love', 'relief', 'gratitude', 'pride'}
EMOTIONS_NEGATIVE = {'annoyance', 'disappointment', 'disgust', 'anger', 'sadness', 'disapproval', 'fear', 'embarrassment', 'nervousness', 'confusion', 'grief', 'remorse'}


def classify(review):
    classifier_output = classifier(review)
    emotion = classifier_output[0][0]['label']  # type: ignore
    if emotion in EMOTIONS_NEGATIVE:
        return 0, emotion
    if emotion in EMOTIONS_POSITIVE:
        return 1, emotion
    raise ValueError(f'Unknown emotion: {emotion}')


def streamlit_tab_single():
    text = st.text_area('Review text')
    if not text:
        return
    classifier_output = classifier(text)
    emotion = classifier_output[0][0]['label']  # type: ignore
    result_str = 'negative' if emotion in EMOTIONS_NEGATIVE else 'positive'
    st.text(f'Review is {result_str}')
    st.text(f'Review has the following primary emotion: {emotion}')
    with st.expander("See full results"):
        labels = list(classifier_output[0] | select(lambda x: x['label']))  # type: ignore
        scores = list(classifier_output[0] | select(lambda x: x['score']))  # type: ignore
        df = pd.DataFrame(data=scores, index=pd.Index(labels, name='emotion'), columns=['score'])
        df['color'] = df.apply(lambda x: 'red' if x.name in EMOTIONS_NEGATIVE else 'green', axis=1)
        st.plotly_chart(px.bar(df, y='score', color='color', color_discrete_map="identity"))
        st.text(pprint.pformat(classifier_output))


def streamlit_tab_dataset():
    data_url = st.text_input("URL or path of .csv file")
    if not data_url:
        return
    data = pd.read_csv(data_url)
    data[['rating', 'emotion']] = data.apply(lambda x: classify(x['text']), axis=1, result_type='expand')
    st.dataframe(data)
    csv_data = data.to_csv(index=False).encode('utf-8')
    st.download_button('Download dataset', csv_data, 'output.csv', 'text/csv')
    st.plotly_chart(px.pie(data, 'rating', title='Ratings', color='rating', color_discrete_map={0: 'red', 1: 'green'}))
    st.plotly_chart(px.pie(data, 'emotion', title='Emotions'))
    

def streamlit_main():
    tab_single, tab_dataset = st.tabs(['Single review', 'Dataset'])
    with tab_single:
        streamlit_tab_single()
    with tab_dataset:
        streamlit_tab_dataset()


streamlit_main()
# try:
#     streamlit_main()
# except su.StreamlitEndRunException:
#     pass
