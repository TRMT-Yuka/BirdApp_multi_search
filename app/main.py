from fastapi import FastAPI,UploadFile,File
from typing import List,Dict
import difflib
import librosa
from collections import defaultdict
from pykakasi import kakasi
import openai
from dotenv import load_dotenv

import torch
from transformers import Wav2Vec2ForPreTraining,Wav2Vec2Processor,BertModel,BertJapaneseTokenizer,BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
import json
import pickle
import csv
import os


app = FastAPI()

# =============知識グラフ
nodes=dict()
p2c=defaultdict(list)
with open('data/all_BirdDBnode.tsv', mode='r', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter = '\t'):
        nodes[row["id"]] = row
        p2c[row["parent_taxon"]].append(row["id"])
print("knowledge data loading is complete !")


# =============音声モデル
w2v2 = Wav2Vec2ForPreTraining.from_pretrained("model/wav2vec2-bird-jp-all")
print("sound model loading is complete !")


# =============音声埋め込み
with open('data/sound_vecs.json') as f:
    sound_vecs = json.load(f)
print("sound vec data loading is complete !")


# =============言語モデル
# # ローカルから日英BERTのモデル・トークナイザーを読み込み
# en_model = BertModel.from_pretrained('model/en_model')
# en_tokenizer = BertTokenizer.from_pretrained('model/en_tokenizer')
# ja_model = BertModel.from_pretrained('model/ja_model')
# ja_tokenizer = BertJapaneseTokenizer.from_pretrained('model/ja_tokenizer')
# print("language model loading is complete !")

# # リモートから日英BERTのモデル・トークナイザーを読み込み
# en_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# en_model = BertModel.from_pretrained('bert-base-uncased')
# ja_tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
# ja_model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

# # Multilingual BERTのモデル・トークナイザーを読み込み
# tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
# model = BertModel.from_pretrained("bert-base-multilingual-cased")

# =============言語埋め込み

# with open('data/en_name_vecs.bin','rb') as bf:
#     en_name_vecs = pickle.load(bf)
# print("en_name_vecs data loading is complete !")

# with open('data/ja_name_vecs.bin','rb') as bf:
#     ja_name_vecs = pickle.load(bf)
# print("ja_name_vecs data loading is complete !")

# with open('data/en_aliases_vecs_all.bin','rb') as bf:
#     en_aliases_vecs = pickle.load(bf)
# print("en_aliases_vecs_all data loading is complete !")

# with open('data/ja_aliases_vecs.bin','rb') as bf:
#     ja_aliases_vecs = pickle.load(bf)
# print("ja_aliases_vecs data loading is complete !")
# print("language vec data loading is complete !")

# =============queryとwordを引数にとり，類似度を返す関数=>これはBERTにしなければならない
def raito(query,word):
    raito = difflib.SequenceMatcher(None, query, word).ratio()
    return raito

# =============queryとwordを引数にとり，BERT類似度を返す関数

# def raito_bert_en(q_vec, word, aliases):
#     # 文をBERTの分散表現に変換する
#     tokenizer = en_tokenizer
#     model = en_model
    
#     if aliases == False:w_vec = en_name_vecs[word]# wordの分散表現
#     else:w_vec = en_aliases_vecs[word]

#     similarity = cosine_similarity(q_vec.unsqueeze(0).numpy(), w_vec.unsqueeze(0).numpy())
#     return similarity[0][0]# Cosine類似度


# def raito_bert_ja(q_vec, word, aliases):
#     # 文をBERTの分散表現に変換する
#     tokenizer = ja_tokenizer
#     model = ja_model
    
#     if aliases == False:w_vec = ja_name_vecs[word]
#     else:w_vec = ja_aliases_vecs[word]


#     similarity = cosine_similarity(q_vec.unsqueeze(0).numpy(), w_vec.unsqueeze(0).numpy())
#     return similarity[0][0]# Cosine類似度


# 日英両対応
# def raito_bert(q_vec, word, en, aliases):
#     # 文をBERTの分散表現に変換する
#     if en == True:
#         tokenizer = en_tokenizer
#         model = en_model
        
#         if aliases == False:w_vec = en_name_vecs[word]# wordの分散表現
#         else:w_vec = en_aliases_vecs[word]
            
#     else:
#         tokenizer = ja_tokenizer
#         model = ja_model
        
#         if aliases == False:w_vec = ja_name_vecs[word]
#         else:w_vec = ja_aliases_vecs[word]


#     similarity = cosine_similarity(q_vec.unsqueeze(0).numpy(), w_vec.unsqueeze(0).numpy())
#     return similarity[0][0]# Cosine類似度


# ============= id=>必要な項目のみを含む自身，親，子の辞書を返す関数

def small_d(d):
    type(d)
    if d != None:
        small_d = {"en_name":d["en_name"],
                    "ja_name":d["ja_name"],
                    "en_aliases":d["en_aliases"],
                    "ja_aliases":d["ja_aliases"],
                    "img_urls":d["img_urls"],
                    "taxon_name":d["taxon_name"]
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


# ============= 2つのnumpy.arrayデータの類似度算出関数
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


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


# ============= ChatGPT応答用関数
# OpenAI APIキーを初期化
load_dotenv()
openai.api_key = os.getenv("API_KEY")
debug_mode = os.getenv("DEBUG")

# ChatGPTに質問を送信する関数
def ask_gpt3(question, max_tokens=100):
    bird_prompt = "下記のデータからわかる情報を日本語の話し言葉で表現してください。回答は「検索結果は以下の通りです：」の文言より開始してください。"
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=bird_prompt+f"{question}\n",
        max_tokens=max_tokens,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text.strip()

# =============# 自然言語クエリに最も近いnameを検索,対応するノードのidを取得=>自身，親，子を辞書で返す

# @app.get("/en_search")# 英語BERT検索
# def search_adjacent_nodes(query: str) -> Dict:
#     max_in_wikidata = 0.0# Wikidata内で最大の類似度格納変数
#     max_id_in_wikidata = None# Wikidata内で最大の類似度のID格納変数

#     #英語名,およびそのエイリアスとクエリとの類似度
#     for node_id,node in nodes.items():
#         # r_in_node: rait in node,該当ノードに含まれる関連語全般とクエリの類似度を格納
#         r_in_node = set()

#         tokens = en_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             en_model.eval()
#             output = en_model(**tokens)
#         q_vec = output.last_hidden_state[0][0]  # queryの分散表現

#         r_in_node.add(raito_bert_en(q_vec,node["en_name"],aliases=False))
        
#         if isinstance(node["en_aliases"], dict):
#             for k,v in node["en_aliases"].items():
#                 r_in_node.add(raito_bert_en(q_vec,v,aliases=True))
        
#         if max(r_in_node) != 0.0:
#             if max(r_in_node) > max_in_wikidata:
#                 max_in_wikidata = max(r_in_node)
#                 max_id_in_wikidata = node_id

#     if max_id_in_wikidata!=None:
#         return id2ans(max_id_in_wikidata)
#     else:
#         return None

# @app.get("/ja_search")# BERT検索
# def search_adjacent_nodes(query: str) -> Dict:
#     max_in_wikidata = 0.0# Wikidata内で最大の類似度格納変数
#     max_id_in_wikidata = None# Wikidata内で最大の類似度のID格納変数

#     print(1)
#     #日本語名,およびそのエイリアスとクエリとの類似度
#     for node_id,node in nodes.items():
#         # r_in_node: rait in node,該当ノードに含まれる関連語全般とクエリの類似度を格納
#         r_in_node = set()

#         tokens = ja_tokenizer(query, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             ja_model.eval()
#             output = ja_model(**tokens)
#         q_vec = output.last_hidden_state[0][0]  # queryの分散表現

#         r_in_node.add(raito_bert_ja(q_vec,node["ja_name"],aliases=False))
        
#         if isinstance(node["ja_aliases"], dict):
#             for k,v in node["ja_aliases"].items():
#                 r_in_node.add(raito_bert_ja(q_vec,v,aliases=True))

#         if max(r_in_node) != 0.0:
#             if max(r_in_node) > max_in_wikidata:
#                 max_in_wikidata = max(r_in_node)
#                 max_id_in_wikidata = node_id
#     print(2)     
#     if max_id_in_wikidata!=None:
#         return id2ans(max_id_in_wikidata)
#     else:
#         return None

# @app.get("/search")# BERT日英検索（要入力言語判定）
# def search_adjacent_nodes(query: str) -> Dict:
#     # Wikidata内で最大の類似度格納変数
#     max_in_wikidata = 0.0
#     # Wikidata内で最大の類似度のID格納変数
#     max_id_in_wikidata = None
#     for node_id,node in nodes.items():
#         #英語名,日本語名,英語名・日本語名のエイリアスとの類似のクエリとの類似
#         # r_in_node: rait in node,該当ノードに含まれる関連語全般とクエリの類似度を格納
#         r_in_node = set()


#         tokens = tokenizer(query, return_tensors="pt", padding=True, truncation=True)
#         with torch.no_grad():
#             model.eval()
#             output = model(**tokens)
#         q_vec = output.last_hidden_state[0][0]  # queryの分散表現

#         r_in_node.add(raito_bert(q_vec,node["en_name"],en=True,aliases=False)) #途中！
#         r_in_node.add(raito_bert(q_vec,node["ja_name"],en=False,aliases=False))
        
#         if isinstance(node["en_aliases"], dict):
#             for k,v in node["en_aliases"].items():
#                 r_in_node.add(raito_bert(q_vec,v,en=True,aliases=True))
#         if isinstance(node["ja_aliases"], dict):
#             for k,v in node["ja_aliases"].items():
#                 r_in_node.add(raito_bert(q_vec,v,en=False,aliases=True))


#         if max(r_in_node) != 0.0:
#             if max(r_in_node) > max_in_wikidata:
#                 max_in_wikidata = max(r_in_node)
#                 max_id_in_wikidata = node_id

#     if max_id_in_wikidata!=None:
#         return id2ans(max_id_in_wikidata)
#     else:
#         return None

@app.get("/search")# 部分一致検索
def search_adjacent_nodes(query: str) -> Dict:

    query = to_katakana(query)
    # Wikidata内で最大の類似度格納変数
    max_in_wikidata = 0.0
    # Wikidata内で最大の類似度のID格納変数
    max_id_in_wikidata = None

    # id_cos_d
    print(1)
    for node_id,node in nodes.items():
        #英語名,日本語名,英語名・日本語名のエイリアスとの類似のクエリとの類似
        # r_in_node: rait in node,該当ノードに含まれる関連語全般とクエリの類似度を格納
        r_in_node = set()

        r_in_node.add(raito(query,node["en_name"]))
        r_in_node.add(raito(query,node["ja_name"]))
        
        if isinstance(node["en_aliases"], dict):
            for k,v in node["en_aliases"].items():
                r_in_node.add(raito(query,v))
        if isinstance(node["ja_aliases"], dict):
            for k,v in node["ja_aliases"].items():
                r_in_node.add(raito(query,v))


        if max(r_in_node) != 0.0:
            if max(r_in_node) > max_in_wikidata:
                max_in_wikidata = max(r_in_node)
                max_id_in_wikidata = node_id
    print(2)
    if max_id_in_wikidata!=None:
        ans_json = id2ans(max_id_in_wikidata)
        print(ans_json)
        gpt_ans = ask_gpt3(ans_json)
        print("ChatGPT's answer:"+gpt_ans)
        return gpt_ans
    else:
        return None



# =============音声 => 再類似ノード(記事の欠陥により複数あり)・その親と子を含む辞書をリストに格納し返す関数
@app.post("/sound/")
async def create_upload_file(file: UploadFile = File(...)):
    # アップロードされた音声ファイルを保存
    file_path = f"uploaded/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    sound_data,_ = librosa.load(file_path, sr=16000)
    result = w2v2(torch.tensor([sound_data]))
    hidden_vecs = result.projected_states
    input_vecs = np.mean(hidden_vecs[0].cpu().detach().numpy(), axis=0)

    max_cos_sim = 0.0
    max_in_sounddata = None

    id_cos_d = dict()

    for d in sound_vecs:
        cos = cos_sim(input_vecs,d["vector"])
        id_cos_d[d["id"][0]]=cos
        # if cos > max_cos_sim:
        #     max_cos_sim = cos
        #     max_in_sounddata = d

    id_cos_sorted = sorted(id_cos_d.items(), key=lambda x:x[1],reverse=True)
    print(id_cos_sorted)
    print(id_cos_sorted[0:3])

    ans_list = []
    for id_cos_tup in id_cos_sorted[0:3]:
        ans_list.append(id2ans(id_cos_tup[0]))
    print(ans_list)
    return ans_list