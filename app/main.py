from fastapi import FastAPI,UploadFile,File,Form
from typing import List,Dict
import difflib
import librosa
from collections import defaultdict
from pykakasi import kakasi
from dotenv import load_dotenv

import openai

from fastapi import FastAPI,Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse,JSONResponse,RedirectResponse

#鍵関連
from fastapi import Depends, HTTPException
from starlette.middleware.sessions import SessionMiddleware

import numpy as np
import json
import pickle
import csv
import os
import ast
import shutil

import torch
from transformers import Wav2Vec2ForPreTraining,Wav2Vec2Processor
from transformers import BertModel,BertJapaneseTokenizer,BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity


# =============アプリケーション
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
app.add_middleware(SessionMiddleware, secret_key="your_secret_key")

# # =============知識グラフ
nodes=dict()
p2c=defaultdict(list)
with open('data/all_BirdDBnode.tsv', mode='r', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter = '\t'):
        nodes[row["id"]] = row
        p2c[row["parent_taxon"]].append(row["id"])
print("knowledge data loading...")

# # =============音声モデル
w2v2 = Wav2Vec2ForPreTraining.from_pretrained("model/wav2vec2-bird-jp-all")
print("sound model loading...")

# # =============言語モデル
# # ローカルから日英BERTのモデル・トークナイザーを読み込み
en_model = BertModel.from_pretrained('model/en_model')
en_tokenizer = BertTokenizer.from_pretrained('model/en_tokenizer')
ja_model = BertModel.from_pretrained('model/ja_model')
ja_tokenizer = BertJapaneseTokenizer.from_pretrained('model/ja_tokenizer')
print("language model loading...")

# # =============言語埋め込み
def read_bin(filename):
    with open(filename,'rb') as bf:
        bin_data = pickle.load(bf)
    return bin_data

bid2Gvec = read_bin('data/bid2Gvec.bin')
en_concat_vecs = read_bin('data/en_concat_vecs.bin')
ja_concat_vecs = read_bin('data/ja_concat_vecs.bin')
print("setup is complete!")


# ============= id=>必要な項目のみを含む自身，親，子の辞書を返す関数
def small_d(d):
    if d != None:
        small_d = {"id":d["id"],
                    "en_name":d["en_name"],
                    "ja_name":d["ja_name"],
                    "en_aliases":d["en_aliases"],
                    "ja_aliases":d["ja_aliases"],
                    "img_urls":d["img_urls"],
                    "taxon_name":d["taxon_name"],
                    "BR_id":d["BirdResearchDB_label01_32k_audio_id"],
                    "JP_id":d["BirdJPBookDB__data_audio_id"]
                    }
    else:
        small_d == None
    return small_d

def id2ans(myself_id):
    ans = {"myself":dict(),"my_parent":None,"my_children":None}
    myself_d  = nodes[myself_id]
    parent_id = nodes[myself_id]["parent_taxon"]
    parent_d  = nodes[parent_id]

    ans["myself"] = small_d(myself_d)
    # 指定したノードidのWikiData隣接ノードを取得
    if parent_id in nodes:
        ans["my_parent"] = small_d(parent_d)
    if myself_id in p2c:
        ans["my_children"] = [small_d(nodes[chile_id]) for chile_id in p2c[myself_id]]
    return ans

# ============= 2つのListのCos類似度算出関数
def cos_sim(v1, v2):
    dot_product = sum(i * j for i, j in zip(v1, v2))
    norm_v1 = sum(i**2 for i in v1) ** 0.5
    norm_v2 = sum(i**2 for i in v2) ** 0.5
    return dot_product / (norm_v1 * norm_v2)

# ============= 漢字および平仮名をカタカナに修正する関数
def to_katakana(text):
    # kakasiオブジェクトを作成
    kakasi_instance = kakasi()
    kakasi_instance.setMode("J", "K")  # J（漢字）をH（ひらがな）に変換
    kakasi_instance.setMode("H", "K")  # H（ひらがな）をK（カタカナ）に変換
    
    # カタカナに変換
    conv = kakasi_instance.getConverter()
    katakana_text = conv.do(text)
    return katakana_text

# =============OpenAI APIキー手動入力画面設定
@app.middleware("http")
async def some_middleware(request: Request, call_next):
    response = await call_next(request)
    session = request.cookies.get('session')
    if session:
        response.set_cookie(key='session', value=request.cookies.get('session'), httponly=True)
    return response

# ============= # ChatGPT質問送信関数
def ask_gpt3(question,api_key,max_tokens=2600):
    # bird_prompt = "次のjsonがどのような情報を持っているかを「お探しの鳥はこれかも：」から始まる簡潔な話し言葉で伝えてください。"
    # bird_prompt = "このjsonデータについて，何が分かりますか？"
    bird_prompt = "このデータを基にこの鳥について解説して:"

    load_dotenv()
    openai.api_key = api_key
    debug_mode = True
    json_string = json.dumps(question)

    response = openai.Completion.create(
        engine="text-davinci-003",
        # prompt=bird_prompt+f"{question}\n",
        prompt=bird_prompt+json_string,
        max_tokens=max_tokens,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# =============# 辞書→WebサイトHTML生成関数
def aliases_str(d_aliases):
    if d_aliases == "{}":
        return ""
    else:
        # print("d_aliases:",d_aliases)
        d_aliases = ast.literal_eval(d_aliases)
        aliases = ""
        for k,v in d_aliases.items():
            aliases = aliases+v+"/"

        aliases = aliases[:-1]
        if aliases != "":
            aliases = "("+aliases+")"
            
        return aliases

def imgs_list(d_img_urls):
    img_urls = []
    if d_img_urls == "{}":
        pass
    else:
        d_img_urls = json.loads(d_img_urls.replace("'",'"'))
        for k,v in d_img_urls.items():
            img_urls.append(v)
    return img_urls

def d4html_empty(self_or_parent,n):#word検索ではnは無し，sound検索では_1,_2,_3
    new_d ={
    self_or_parent+"_taxon_name"+n:"",
    self_or_parent+"_ja_name"+n:"",
    self_or_parent+"_ja_aliases"+n:[],
    self_or_parent+"_en_name"+n:"",
    self_or_parent+"_en_aliases"+n:[],
    self_or_parent+"_link"+n:"",
    self_or_parent+"_imgs_list"+n:[]
    }
    return new_d

def d4html(myself_d,self_or_parent,n):#word検索ではnは無し，sound検索では_1,_2,_3
    new_d ={
    self_or_parent+"_taxon_name"+n:myself_d["taxon_name"],
    self_or_parent+"_ja_name"+n:myself_d["ja_name"],
    self_or_parent+"_ja_aliases"+n:aliases_str(myself_d["ja_aliases"]),
    self_or_parent+"_en_name"+n:myself_d["en_name"],
    self_or_parent+"_en_aliases"+n:aliases_str(myself_d["en_aliases"]),
    self_or_parent+"_link"+n:"https://www.wikidata.org/wiki/"+myself_d["id"],
    self_or_parent+"_imgs_list"+n:imgs_list(myself_d["img_urls"])
    }
    return new_d

def self_d4html(myself_d,n):#word検索ではnは無し，sound検索では_1,_2,_3
    new_d ={"self_taxon_name"+n:myself_d["taxon_name"],
    "self_ja_name"+n:myself_d["ja_name"],
    "self_ja_aliases"+n:aliases_str(myself_d["ja_aliases"]),
    "self_en_name"+n:myself_d["en_name"],
    "self_en_aliases"+n:aliases_str(myself_d["en_aliases"]),
    "self_link"+n:"https://www.wikidata.org/wiki/"+myself_d["id"],
    "self_imgs_list"+n:imgs_list(myself_d["img_urls"])
    }
    print(new_d)
    return new_d

def parent_d4html(parent_d,n):#word検索ではnは無し，sound検索では_1,_2,_3
    new_d ={"parent_taxon_name"+n:parent_d["taxon_name"],
    "parent_ja_name"+n:parent_d["ja_name"],
    "parent_ja_aliases"+n:aliases_str(parent_d["ja_aliases"]),
    "parent_en_name"+n:parent_d["en_name"],
    "parent_en_aliases"+n:aliases_str(parent_d["en_aliases"]),
    "parent_link"+n:"https://www.wikidata.org/wiki/"+parent_d["id"],
    "parent_imgs_list"+n:imgs_list(parent_d["img_urls"])
    }
    return new_d

#=================================================マルチモーダル検索用関数
def concat_vecs(query,lang,Wikidata_id,sound_file_dir):
    inp_Lvec = [0]*768
    inp_Gvec = [0]*64
    inp_Svec = [0]*256

    if query != None:
        query = to_katakana(query)
        if lang=="en":
            en_tokens = en_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                en_model.eval()
                output = en_model(**en_tokens)
        else:
            ja_tokens = ja_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                ja_model.eval()
                output = ja_model(**ja_tokens)
        inp_Lvec = output.last_hidden_state[0][0].tolist()# queryの分散表現

    if Wikidata_id != None and Wikidata_id in bid2Gvec:
        inp_Gvec = bid2Gvec[Wikidata_id]

    if sound_file_dir != None:
        sound_data,_ = librosa.load(sound_file_dir, sr=16000)
        result = w2v2(torch.tensor([sound_data]))
        hidden_vecs = result.projected_states
        inp_Svec = np.mean(hidden_vecs[0].cpu().detach().numpy(), axis=0)
        inp_Svec = inp_Svec.tolist()
    return inp_Lvec+inp_Gvec+inp_Svec
    
def multi_Search_top3(concat_vecs,input_v):
    id_cos_tpls = []
    for bid,v in concat_vecs:
        t = (bid,cos_sim(input_v,v))
        id_cos_tpls.append(t)

    # タプル2つ目が大きい順にソート
    sorted_list = sorted(id_cos_tpls, key=lambda x: x[1], reverse=True)

    # 1つ目の値が異なるタプルを保持するための辞書
    unique_first_values = {}

    # 上位3つの異なる一つ目の値を取得
    top_three_tuples = []
    for item in sorted_list:
        first_value = item[0]
        if first_value not in unique_first_values:
            unique_first_values[first_value] = item
            top_three_tuples.append(item)
            if len(top_three_tuples) == 3:
                break

    return top_three_tuples

#=================================================マルチモーダル検索
@app.get("/")
def read_root():
    # リダイレクト先のURLを指定してRedirectResponseを作成
    redirect_url = "/multi_search"
    response = RedirectResponse(url=redirect_url)
    return response

@app.get("/multi_search", response_class=HTMLResponse)
async def read_root(request:Request, api_key:str=""):
    api_key = request.session.get("api_key", "Please input your key")
    return templates.TemplateResponse("multi_search.html", {"request": request,"api_key":api_key})

@app.post("/multi_search",response_class=HTMLResponse)
async def multi_search(request:Request,
    api_key:str=Form(...),
    query:str=Form(None),
    lang:str=Form(None),
    Wikidata_id:str=Form(None),
    file: UploadFile = File(None)):  

    try:
        with open("uploaded/"+file.filename, "wb") as f:
            f.write(await file.read())
        sound_file_dir = "uploaded/"+file.filename

    except: 
        sound_file_dir = None

    if query==None and Wikidata_id==None and sound_file_dir==None:
        return templates.TemplateResponse("multi_search.html", 
            {**{"request": request,"api_key":api_key,"message":"入力値がありません",
                "max_cos_in_wikidata_1":"","max_cos_in_wikidata_2":"","max_cos_in_wikidata_3":"",
                "gpt_ans_self_1":"","gpt_ans_parent_1":"","gpt_ans_children_1":"",
                "gpt_ans_self_2":"","gpt_ans_parent_2":"","gpt_ans_children_2":"",
                "gpt_ans_self_3":"","gpt_ans_parent_3":"","gpt_ans_children_3":""},
                **d4html_empty("self","_1"),**d4html_empty("parent","_1"),
                **d4html_empty("self","_2"),**d4html_empty("parent","_2"),
                **d4html_empty("self","_3"),**d4html_empty("parent","_3")
            })

    input_vecs = concat_vecs(query,lang,Wikidata_id,sound_file_dir)

    if lang=="en":
        (id1,cos1),(id2,cos2),(id3,cos3) = multi_Search_top3(en_concat_vecs,input_vecs)
    else:
        (id1,cos1),(id2,cos2),(id3,cos3) = multi_Search_top3(ja_concat_vecs,input_vecs)

    ans_json_1 = id2ans(id1)
    ans_json_2 = id2ans(id2)
    ans_json_3 = id2ans(id3)

    gpt_ans_self_1 = ask_gpt3(ans_json_1["myself"],api_key)
    gpt_ans_parent_1 = ask_gpt3(ans_json_1["my_parent"],api_key)
    gpt_ans_children_1 = ask_gpt3(ans_json_1["my_children"],api_key)

    gpt_ans_self_2 = ask_gpt3(ans_json_2["myself"],api_key)
    gpt_ans_parent_2 = ask_gpt3(ans_json_2["my_parent"],api_key)
    gpt_ans_children_2 = ask_gpt3(ans_json_2["my_children"],api_key)

    gpt_ans_self_3 = ask_gpt3(ans_json_3["myself"],api_key)
    gpt_ans_parent_3 = ask_gpt3(ans_json_3["my_parent"],api_key)
    gpt_ans_children_3 = ask_gpt3(ans_json_3["my_children"],api_key)

    return templates.TemplateResponse("multi_search.html", 
        {**{"request": request,
        "api_key":api_key,
        "messeage":"",
        "max_cos_in_wikidata_1":round(cos1,4),#類似度
        "max_cos_in_wikidata_2":round(cos2,4),
        "max_cos_in_wikidata_3":round(cos3,4),
        "gpt_ans_self_1": gpt_ans_self_1,
        "gpt_ans_parent_1": gpt_ans_parent_1,
        "gpt_ans_children_1": gpt_ans_children_1,
        "gpt_ans_self_2": gpt_ans_self_2,
        "gpt_ans_parent_2": gpt_ans_parent_2,
        "gpt_ans_children_2": gpt_ans_children_2,
        "gpt_ans_self_3": gpt_ans_self_3,
        "gpt_ans_parent_3": gpt_ans_parent_3,
        "gpt_ans_children_3": gpt_ans_children_3},
        **d4html(ans_json_1["myself"],"self","_1"),
        **d4html(ans_json_1["my_parent"],"parent","_1"),
        **d4html(ans_json_2["myself"],"self","_2"),
        **d4html(ans_json_2["my_parent"],"parent","_2"),
        **d4html(ans_json_3["myself"],"self","_3"),
        **d4html(ans_json_3["my_parent"],"parent","_3")
        })