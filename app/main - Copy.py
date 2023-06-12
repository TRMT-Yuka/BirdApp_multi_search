from fastapi import FastAPI
from typing import List,Dict
import csv
import difflib
from collections import defaultdict

app = FastAPI()

# WikiDATAの読込
nodes=dict()
p2c=defaultdict(list)
with open('all_BirdDBnode.tsv', mode='r', newline='', encoding='utf-8') as f:
    for row in csv.DictReader(f, delimiter = '\t'):
        nodes[row["id"]] = row
        p2c[row["parent_taxon"]].append(row["id"])

# queryとwordを引数にとり，類似度を返す関数
def raito(query,word):
    raito = difflib.SequenceMatcher(None, query, word).ratio()
    return raito


@app.get("/search")#途中

def search_adjacent_nodes(query: str) -> List[Dict]:

    # 自然言語クエリに最も近いnameを検索し、対応するノードのidを取得
    max_in_wikidata = 0.0
    max_id_in_wikidata = ""
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

        print(node["en_name"])
        print(r_in_node,"hokaku",max_in_wikidata)
        if max(r_in_node) > max_in_wikidata:
            max_in_wikidata = r_in_node
            max_id_in_wikidata = node_id

    ans = {"myself":nodes[max_id_in_wikidata],"my_parent":dict(),"my_children":dict()}
    
    # 指定したノードidのWikiData隣接ノードを取得
    if node["parent_taxon"] in nodes:
        ans["my_parent"]=nodes[max_id_in_wikidata]
    if node_id in p2c:
        ans["my_children"]=[nodes[each_chile_id] for each_chile_id in p2c[max_id_in_wikidata]]
    ans_nodes.append(ans)

    for ans in ans_nodes:
        print("myself:",ans["myself"]["ja_name"])
        print("my_parent:",ans["my_parent"]["ja_name"])
        for c in ans["my_children"]:
            # print(c)
            print("my_children:",c["ja_name"])
    return ans_nodes

"""
def search_adjacent_nodes(query: str) -> List[Dict]:
    # 自然言語クエリに最も近いnameを検索し、対応するノードのidを取得
    ans_nodes = []
    for node_id,node in nodes.items():

        #英語名,日本語名,英語名・日本語名のエイリアスとの類似のクエリとの類似
        expressions = set()
        expressions.add(node["en_name"])
        expressions.add(node["ja_name"])
        if isinstance(node["en_aliases"], dict):
            for k,v in node["en_aliases"].items():
                expressions.add(v)
        if isinstance(node["ja_aliases"], dict):
            for k,v in node["ja_aliases"].items():
                expressions.add(v)
        max_raito=0
        for word in expressions:
            ratio = difflib.SequenceMatcher(None, query, word).ratio()
            if ratio > max_raito:
                print(word,ratio)
                ans = {"myself":node,"my_parent":dict(),"my_children":dict()}
                
                # 指定したノードidのWikiData隣接ノードを取得
                if node["parent_taxon"] in nodes:
                    ans["my_parent"]=nodes[node["parent_taxon"]]
                if node_id in p2c:
                    ans["my_children"]=[nodes[each_chile_id] for each_chile_id in p2c[node_id]]
                ans_nodes.append(ans)
    for ans in ans_nodes:
        print("myself:",ans["myself"]["ja_name"])
        print("my_parent:",ans["my_parent"]["ja_name"])
        for c in ans["my_children"]:
            # print(c)
            print("my_children:",c["ja_name"])
    return ans_nodes
"""