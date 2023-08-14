from fastapi import FastAPI,UploadFile,File
from typing import List,Dict
import csv
import difflib
from collections import defaultdict
import os
import torch
from transformers import Wav2Vec2ForPreTraining,Wav2Vec2Processor
import librosa
import numpy as np
import json

app = FastAPI()


# =============WikiDATAの読込
nodes=dict()
p2c=defaultdict(list)
with open('data/all_BirdDBnode.tsv', mode='r', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter = '\t'):
        nodes[row["id"]] = row
        p2c[row["parent_taxon"]].append(row["id"])



# =============wav2vec2の読込
w2v2 = Wav2Vec2ForPreTraining.from_pretrained("wav2vec2-bird-jp-all")



# =============vecs(音埋込)の読込
with open('data/vecs.json') as f:
    sound_vecs = json.load(f)



# =============queryとwordを引数にとり，類似度を返す関数=>これはBERTにしなければならない
def raito(query,word):
    raito = difflib.SequenceMatcher(None, query, word).ratio()
    return raito



# ============= id=>自身，親，子を辞書で返す関数
def id2ans(the_id):
    ans = {"myself":dict(),"my_parent":None,"my_children":None}
    ans["myself"] = nodes[the_id]
    # 指定したノードidのWikiData隣接ノードを取得
    if ans["myself"]["parent_taxon"] in nodes:
        ans["my_parent"] = nodes[ans["myself"]["parent_taxon"]]
    if the_id in p2c:
        ans["my_children"] = [nodes[each_chile_id] for each_chile_id in p2c[the_id]]
    return ans



# ============= 2つのnumpy.arrayデータの類似度算出関数
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))



# =============# 自然言語クエリに最も近いnameを検索,対応するノードのidを取得=>自身，親，子を辞書で返す
@app.get("/search")#途中
def search_adjacent_nodes(query: str) -> Dict:
    # Wikidata内で最大の類似度格納変数
    max_in_wikidata = 0.0
    # Wikidata内で最大の類似度のID格納変数
    max_id_in_wikidata = None

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
            # print(node["ja_name"]+": "+str(max(r_in_node)))

            if max(r_in_node) > max_in_wikidata:
                max_in_wikidata = max(r_in_node)
                max_id_in_wikidata = node_id

    if max_id_in_wikidata!=None:
        print(str(id2ans(max_id_in_wikidata))+": "+str(max_in_wikidata))
        return id2ans(max_id_in_wikidata)
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
