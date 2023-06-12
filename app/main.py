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
def search_adjacent_nodes(query: str) -> Dict:

    # 自然言語クエリに最も近いnameを検索し、対応するノードのidを取得
    max_in_wikidata = 0.0
    max_id_in_wikidata = None
    print(2)
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

    ans = {"myself":dict(),"my_parent":dict(),"my_children":List[Dict]}

    if max_id_in_wikidata!=None:
        ans["myself"]=nodes[max_id_in_wikidata]

        # 指定したノードidのWikiData隣接ノードを取得
        if node["parent_taxon"] in nodes:
            ans["my_parent"]=nodes[ans["myself"]["parent_taxon"]]
        if max_id_in_wikidata in p2c:
            ans["my_children"]=[nodes[each_chile_id] for each_chile_id in p2c[max_id_in_wikidata]]

    return ans