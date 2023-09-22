import streamlit as st
from ext_sum_onnx import M_Sum
from abs_sum_onnx import abs_Sum
from underthesea import sent_tokenize
import re
@st.cache_resource
def load_model_ext():
    ext_vi_sum = M_Sum('vi')
    ext_ru_sum = M_Sum('ru')
    ext_en_sum = M_Sum('en')
    ext_ch_sum = M_Sum('ch')
    return ext_vi_sum, ext_ru_sum, ext_en_sum, ext_ch_sum
ext_vi_sum, ext_ru_sum, ext_en_sum, ext_ch_sum = load_model_ext()

@st.cache_resource
def load_model_abs():
    abs_vi_sum = abs_Sum('vietnamese')
    abs_ru_sum = abs_Sum('russian')
    abs_en_sum = abs_Sum('english')
    abs_ch_sum = abs_Sum('chinese')
    return abs_vi_sum, abs_ru_sum, abs_en_sum, abs_ch_sum
abs_vi_sum, abs_ru_sum, abs_en_sum, abs_ch_sum = load_model_abs()


#streamlit
st.title("AIA M_Summarizer")
st.subheader("Paste any article in text area below and click on the 'Summarize Text' button to get the summarized textual data.")
st.subheader("Or enter a url from news stites such as dantri.com and click on the 'Import Url' button to get the summarized.")
st.subheader('This application is powered by Minggz.')
option = st.selectbox(
    'Choosing language: ',
    ('Vietnamese', 'Chinese', 'English','Russian'))
st.write('You selected:', option)
st.write('Paste your copied data or news url here ...')
txt = st.text_area(label='Input', placeholder='Try me', max_chars=5000, height=50)

news = {'title':'',
            'description':'',
            'paras':txt}

print(news)
summary_length = st.select_slider(label='Summary Length', options=['Extreme Short','Short', 'Medium', 'Long', 'Extreme Long'])
st.write(summary_length)
col1, col2 = st.columns([2,6],gap='large')
with col1:
    ext_button = st.button(label='Extractive Text Summarization')
with col2:
    abs_button = st.button(label='Abstractive Text Summarization')

sum_len_dict = {'Extreme Short':0.2,'Short':0.4, 'Medium':0.6, 'Long':0.8, 'Extreme Long':0.9}
k = sum_len_dict[summary_length]

if option == 'Vietnamese':
    if ext_button :
        result = ext_vi_sum.sum_main(news,k)
        st.write(result)
        # print(vi_sum.pretrained)
    if abs_button:
        if '\n\n' in txt:
            n = len(txt.split('\n\n'))
            num_sent = round(n*k)
            result = abs_vi_sum.summary_n(txt, num_sent)
            st.write(result)
        else:    
            n = len(sent_tokenize(txt))
            num_sent = round(n*k)
            result = abs_vi_sum.summary_dot(txt, num_sent)
            st.write(result)

elif option == 'English':
    if ext_button :
        result = ext_en_sum.sum_main(news,k)
        st.write(result)
    if abs_button:
        if '\n\n' in txt:
            n = len(txt.split('\n\n'))
            num_sent = round(n*k)
            result = abs_en_sum.summary_n(txt, num_sent)
            st.write(result)
        else:    
            n = len(sent_tokenize(txt))
            num_sent = round(n*k)
            result = abs_en_sum.summary_dot(txt, num_sent)
            st.write(result)


elif option == 'Russian':
    if ext_button :
        result = ext_ru_sum.sum_main(news,k)
        st.write(result)
    if abs_button:
        if '\n\n' in txt:
            n = len(txt.split('\n\n'))
            num_sent = round(n*k)
            result = abs_ru_sum.summary_n(txt, num_sent)
            st.write(result)
        else:    
            n = len(sent_tokenize(txt))
            num_sent = round(n*k)
            result = abs_ru_sum.summary_dot(txt, num_sent)
            st.write(result)

else :
    if ext_button :
        result = ext_ch_sum.sum_main(news,k)
        st.write(result)
    if abs_button:
        if '\n\n' in txt:  
            n = len(txt.split('\n\n'))
            num_sent = round(n*k)
            result = abs_ru_sum.summary_n(txt, num_sent)
            st.write(result)
        else:  
            list_sent = [s.replace('\u3000',' ').strip() for s in re.split('[。！？；]', txt)]
            n = len(list_sent)
            num_sent = round(n*k)
            result = abs_ch_sum.summary(txt, num_sent)
            st.write(result)
    