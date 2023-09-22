from ext_sum_onnx import M_Sum
from abs_sum_onnx import abs_Sum
from underthesea import sent_tokenize
from fastapi import FastAPI
import uvicorn
from typing import List, Optional, Union
from pydantic import BaseModel, Field
from starlette.middleware.cors import CORSMiddleware
import re


def load_model_ext():
    ext_vi_sum = M_Sum('vi')
    ext_ru_sum = M_Sum('ru')
    ext_en_sum = M_Sum('en')
    ext_ch_sum = M_Sum('ch')
    return ext_vi_sum, ext_ru_sum, ext_en_sum, ext_ch_sum


ext_vi_sum, ext_ru_sum, ext_en_sum, ext_ch_sum = load_model_ext()

print('--------------- Done Loading EXT ---------------')

def load_model_abs():
    abs_vi_sum = abs_Sum('vietnamese')
    abs_ru_sum = abs_Sum('russian')
    abs_en_sum = abs_Sum('english')
    abs_ch_sum = abs_Sum('chinese')
    return abs_vi_sum, abs_ru_sum, abs_en_sum, abs_ch_sum


abs_vi_sum, abs_ru_sum, abs_en_sum, abs_ch_sum = load_model_abs()

print('--------------- Done Loading ABS ---------------')


app = FastAPI()
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_methods=["*"],
                   allow_headers=["*"])


class Document(BaseModel):
    lang: str = Field(...)
    title: str = Field(...)
    description: str = Field(...)
    paras: str = Field(...)
    k : float = Field(...)


@app.get("/")
async def hello():
    return {"Welcome to": " Multilingual Summarization"}


@app.post("/ext")
async def ext_sum(news: Document):
    news = dict(news)

    lang = news['lang']
    k = news['k']
    news.pop('lang')
    print(news)
    print(type(news))
    if lang == 'vi':
        summary = ext_vi_sum.sum_main(news, k)
    elif lang == 'ch':
        summary = ext_ch_sum.sum_main(news, k)
    elif lang == 'ru':
        summary = ext_ru_sum.sum_main(news, k)
    else:
        summary = ext_en_sum.sum_main(news, k)
    return {'Extractive summarization': summary}


@app.post("/abs")
async def abs_sum(news: Document):
    news = dict(news)
    k = news['k']
    lang = news['lang']
    txt = news['paras']
    if lang == 'vi':
        if '\n\n' in txt:
            n = len(txt.split('\n\n'))
            num_sent = round(n*k)
            result = abs_en_sum.summary_n(txt, num_sent)

        else:
            n = len(sent_tokenize(txt))
            num_sent = round(n*k)
            result = abs_en_sum.summary_dot(txt, num_sent)
    elif lang == 'ch':
        if '\n\n' in txt:
            n = len(txt.split('\n\n'))
            num_sent = round(n*k)
            result = abs_ru_sum.summary_n(txt, num_sent)
        else:
            list_sent = [s.replace('\u3000', ' ').strip()
                         for s in re.split('[。！？；]', txt)]
            n = len(list_sent)
            num_sent = round(n*k)
            result = abs_ch_sum.summary(txt, num_sent)
    elif lang == 'ru':
        if '\n\n' in txt:
            n = len(txt.split('\n\n'))
            num_sent = round(n*k)
            result = abs_ru_sum.summary_n(txt, num_sent)
        else:
            n = len(sent_tokenize(txt))
            num_sent = round(n*k)
            result = abs_ru_sum.summary_dot(txt, num_sent)
    else:
        if '\n\n' in txt:
            n = len(txt.split('\n\n'))
            num_sent = round(n*k)
            result = abs_en_sum.summary_n(txt, num_sent)
        else:
            n = len(sent_tokenize(txt))
            num_sent = round(n*k)
            result = abs_en_sum.summary_dot(txt, num_sent)
    return {'Abstractive Summarization': result}


if __name__ == '__main__':
    uvicorn.run("sum_api:app", host="localhost", port=4051, reload=True)