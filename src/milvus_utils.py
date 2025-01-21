from flask import Blueprint, request, jsonify

from flask import  request, jsonify
from langchain_community.embeddings import ZhipuAIEmbeddings
from flask_cors import CORS
from pymilvus import FieldSchema, CollectionSchema, DataType, connections,Collection,MilvusClient,AnnSearchRequest,WeightedRanker
import os,json
from dotenv import load_dotenv, find_dotenv
from zhipuai import ZhipuAI

 # 设置Milvus服务器的连接参数
MILVUS_HOST = 'api.caritas.pro'  # 替换为实际的服务器IP地址
MILVUS_PORT = 19530  # 通常Milvus默认监听此端口，若服务器端有修改，请使用修改后的端口
# 建立与Milvus服务器的连接（Milvus v2.4.x方式）
connections.connect(alias="default", host = MILVUS_HOST, port = MILVUS_PORT)
client = MilvusClient(uri=f"tcp://{MILVUS_HOST}:{MILVUS_PORT}")

# 加载Embedding模型
load_dotenv(find_dotenv())
api_key = os.environ["ZHIPU_API_key"]
embeddings = ZhipuAIEmbeddings(
    model="embedding-3",
    api_key=api_key
)


# 获取智普的客户端
zhipu_client = ZhipuAI(api_key = api_key)  # 请填写您自己的APIKey

# 假设使用的模型生成的向量维度为1152，不同模型维度可能不同，需根据实际调整
dim = 2048
# 设置批次插入的数据行
batch_size = 100


#配置蓝图
milvus_utils_routes = Blueprint('milvus_utils_routes', __name__)

@milvus_utils_routes.route('/create_collection', methods=['POST'])
def create_collection_endpoint():
    """
    创建集合的接口端点

    :return: 创建结果信息
    """
    try:
        collection = create_collection()
        # print("collection",collection)
        return jsonify({"message": "集合创建成功", "collection_name": collection['collection_name']})
    except Exception as e:
        return jsonify({"message": "集合创建失败", "error": str(e)})


@milvus_utils_routes.route('/create_index', methods=['POST'])
def create_index_endpoint():
    """
    为集合创建索引的接口端点

    :param collection_name: 要创建索引的集合名称
    :return: 创建索引结果信息
    """
    collection_name = request.json.get('collection_name')
    field_name_list = request.json.get('field_name_list')
    try:
        create_index(collection_name,field_name_list)
        return jsonify({"message": "索引创建成功", "collection_name": collection_name})
    except Exception as e:
        return jsonify({"message": "索引创建失败", "error": str(e)})


@milvus_utils_routes.route('/insert_data', methods=['POST'])
def insert_data_endpoint():
    """
    插入数据到集合的接口端点
    :param collection_name: 要插入数据的集合名称
    :param json_data: 要插入的JSON格式数据
    :return: 插入数据结果信息
    """
    collection_name = request.json.get('collection_name')
    json_data = request.json.get('json_data')
    try:
        res = insert_data(collection_name, json_data)
        ids_list = list(res['ids'])
        res['ids'] = ids_list
        return jsonify({"message": "数据插入成功", "collection_name": collection_name, "res":res})
    except Exception as e:
        return jsonify({"message": "数据插入失败", "error": str(e)})


@milvus_utils_routes.route('/batch_insert_data', methods=['POST'])
def batch_insert__data_endpoint():
    """
    倒插插入数据到集合的接口端点
    :param collection_name: 要插入数据的集合名称
    """
    collection_name = request.json.get('collection_name')
    try:
        batch_insert_data(collection_name)
        return jsonify({"message": "数据批量插入成功", "collection_name": collection_name})
    except Exception as e:
        return jsonify({"message": "数据批量插入失败", "error": str(e)})


@milvus_utils_routes.route('/delete_data', methods=['POST'])
def delete_data_endpoint():
    """
    从集合中删除数据的接口端点

    :param collection_name: 要删除数据的集合名称
    :param record_id: 要删除的记录ID
    :return: 删除数据结果信息
    """
    collection_name = request.json.get('collection_name')
    record_id = request.json.get('record_id')
    try:
        res = delete_data(collection_name, record_id)
        return jsonify({"message": "数据删除成功", "collection_name": collection_name, "res":res})
    except Exception as e:
        return jsonify({"message": "数据删除失败", "error": str(e)})


@milvus_utils_routes.route('/update_data', methods=['POST'])
def update_data_endpoint():
    """
    更新集合中数据的接口端点
    :param collection_name: 要更新数据的集合名称
    :param updated_data: 包含更新后数据的字典
    :return: 更新数据结果信息
    """
    collection_name = request.json.get('collection_name')
    updated_data = request.json.get('updated_data')
    try:
        res = update_data(collection_name, updated_data)
        ids_list = list(res['ids'])
        res['ids'] = ids_list
        return jsonify({"message": "数据更新成功", "collection_name": collection_name, "res":res})
    except Exception as e:
        return jsonify({"message": "数据更新失败", "error": str(e)})

## 方法 ##

def client_set_load(collection_name):
        load_state = client.get_load_state(collection_name).get("state")
        if load_state != 'Loaded':
            client.load_collection(collection_name)


def create_collection():
    """
    创建Milvus集合

    :return: 创建好的集合对象
    """
    # Article字段
    # 定义字段模式，考虑到要插入所有字段
    article_field1 = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True)
    article_field2 = FieldSchema(name="vector_content", dtype=DataType.FLOAT_VECTOR, dim=dim)
    article_field3 = FieldSchema(name="vector_title", dtype=DataType.FLOAT_VECTOR, dim=dim)
    article_field4 = FieldSchema(name="vector_question", dtype=DataType.FLOAT_VECTOR, dim=dim)
    article_field5 = FieldSchema(name="question", dtype= DataType.VARCHAR, max_length=10000)
    article_field6 = FieldSchema(name="zhihu_link", dtype= DataType.VARCHAR, max_length=10000)
    article_field7 = FieldSchema(name="author", dtype= DataType.VARCHAR, max_length=10000)
    article_field8 = FieldSchema(name="favorites_count", dtype= DataType.INT64)
    article_field9 = FieldSchema(name="last_update", dtype= DataType.VARCHAR, max_length=10000)
    article_field10 = FieldSchema(name="links", dtype= DataType.VARCHAR, max_length=10000)
    article_field11 = FieldSchema(name="title", dtype= DataType.VARCHAR, max_length=10000)
    article_field12 = FieldSchema(name="tags", dtype= DataType.VARCHAR, max_length=10000)
    article_field13 = FieldSchema(name="content", dtype= DataType.VARCHAR, max_length=65535)
    article_field14 = FieldSchema(name="read_count", dtype= DataType.INT64)

    # Excerpt字段
    #定义字段模式，考虑到要插入所有字段
    excerpt_field1 = FieldSchema(name="id", dtype=DataType.INT64,is_primary=True)
    excerpt_field2 = FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    excerpt_field3 = FieldSchema(name="end", dtype=DataType.INT64)
    excerpt_field4 = FieldSchema(name="end_index", dtype=DataType.INT64)
    excerpt_field5 = FieldSchema(name="excerpt_number", dtype=DataType.INT64)
    excerpt_field6 = FieldSchema(name="start", dtype=DataType.INT64)
    excerpt_field7 = FieldSchema(name="create_time", dtype= DataType.VARCHAR, max_length=10000)
    excerpt_field8 = FieldSchema(name="start_index", dtype=DataType.INT64)
    excerpt_field9 = FieldSchema(name="quote", dtype=DataType.VARCHAR, max_length=65535)
    excerpt_field10 = FieldSchema(name="article_id", dtype=DataType.INT64)
    excerpt_field11 = FieldSchema(name="article_question", dtype= DataType.VARCHAR, max_length=10000)
    excerpt_field12 = FieldSchema(name="article_zhihu_link", dtype= DataType.VARCHAR, max_length=10000)
    excerpt_field13 = FieldSchema(name="article_author", dtype= DataType.VARCHAR, max_length=10000)
    excerpt_field14 = FieldSchema(name="article_favorites_count", dtype= DataType.INT64)
    excerpt_field15 = FieldSchema(name="article_last_update", dtype= DataType.VARCHAR, max_length=10000)
    excerpt_field16 = FieldSchema(name="article_links", dtype= DataType.VARCHAR, max_length=10000)
    excerpt_field17 = FieldSchema(name="article_title", dtype= DataType.VARCHAR, max_length=10000)
    excerpt_field18 = FieldSchema(name="article_tags", dtype= DataType.VARCHAR, max_length=10000)
    excerpt_field19 = FieldSchema(name="article_content", dtype= DataType.VARCHAR, max_length=65535)
    excerpt_field20 = FieldSchema(name="article_read_count", dtype= DataType.INT64)

    # 创建Article集合模式
    article_schema = CollectionSchema(fields=[article_field1, article_field2, article_field3, article_field4, article_field5, article_field6, article_field7, article_field8, article_field9, article_field10, article_field11, article_field12, article_field13, article_field14],
                              description="包含所有字段的示例集合描述")
    # 创建Excerpt集合模式
    excerpt_schema = CollectionSchema(fields=[excerpt_field1, excerpt_field2, excerpt_field3, excerpt_field4, excerpt_field5, excerpt_field6, excerpt_field7, excerpt_field8, excerpt_field9, excerpt_field10, excerpt_field11, excerpt_field12, excerpt_field13, excerpt_field14, excerpt_field15, excerpt_field16, excerpt_field17, excerpt_field18, excerpt_field19, excerpt_field20],
                              description="包含所有字段的示例集合描述")

    # 判断Article集合（如果存在则删除）
    article_collection_name = "article_collection"
    if client.has_collection(article_collection_name):
        client.drop_collection(article_collection_name)
    # 创建Article集合
    client.create_collection(collection_name=article_collection_name, schema=article_schema)
    #创建Article集合的索引
    create_index(article_collection_name,['vector_content','vector_title','vector_question'])

    # 判断Excerpt集合（如果存在则删除）
    excerpt_collection_name = "excerpt_collection"
    if client.has_collection(excerpt_collection_name):
        client.drop_collection(excerpt_collection_name)
    # 创建Excerpt集合
    client.create_collection(collection_name=excerpt_collection_name, schema=excerpt_schema)
    #创建Excerpt集合的索引
    create_index(excerpt_collection_name,['vector'])
    
    return client.describe_collection(article_collection_name)


def create_index(collection_name, field_name_list):
        """
        为指定的Milvus集合的vector字段创建索引

        :param collection_name: 集合名称
        :param field_name_list: 字段名称列表
        :return: 无
        """
        for field_name in field_name_list:
            index_params = {
                "metric_type": "COSINE",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 128}
            }
            client.create_index(
                collection_name=collection_name,
                field_name=field_name,
                index_params=index_params,
                index_name=field_name + "_index"
            )


def insert_data(collection_name, json_data):
        """
        向指定的Milvus集合插入数据

        :param collection: Milvus集合对象
        :param json_data: 包含要插入数据的JSON格式数据
        :return: 无
        """
        # 提取article字段
        article_rel_id = json_data["id"]
        article_question = json_data["question"] if 'question' in json_data else ''
        article_zhihu_link = json_data["zhihu_link"]
        article_author = json_data["author"]
        article_favorites_count = json_data["favorites_count"]
        article_last_update = json_data["last_update"]
        article_links = json_data["links"] if 'links' in json_data else ''
        article_title = json_data["title"] if 'title' in json_data else ''
        article_tags = json_data["tags"] if 'tags' in json_data else ''
        article_read_count = json_data["read_count"]
        article_content = json_data["content"]
        # 对相关文本字段进行向量化
        article_vector_content = embeddings.embed_query(article_content)
        article_vector_title = embeddings.embed_query(article_title)
        article_vector_question = embeddings.embed_query(article_question)

        # 准备要插入的单条article数据字典
        article_data_to_insert = {
            "id": article_rel_id,
            "vector_content": article_vector_content, 
            "vector_title": article_vector_title, 
            "vector_question": article_vector_question, 
            "question": article_question,
            "zhihu_link": article_zhihu_link,
            "author": article_author,
            "favorites_count": article_favorites_count,
            "last_update": article_last_update,
            "links": article_links,
            "title": article_title,
            "tags": article_tags,
            "content": article_content,
            "read_count": article_read_count
        }
        # 先加载集合
        client_set_load(collection_name)
        # 插入单条数据到集合
        res = client.insert(collection_name, article_data_to_insert)
        client.refresh_load(collection_name)

        if "excerptCacheList" in json_data:
            excerpt_data_to_insert_list =[]
            for excerptCache in json_data['excerptCacheList']:
                # 提取Excerpt字段
                excerpt_rel_id = excerptCache["id"]
                excerpt_article_id = excerptCache["article_id"]
                excerpt_quote = excerptCache["quote"]
                excerpt_start_index = excerptCache["start_index"]
                excerpt_create_time = excerptCache["create_time"]
                excerpt_start = excerptCache["start"]
                excerpt_excerpt_number = excerptCache["excerpt_number"]
                excerpt_end_index = excerptCache["end_index"]
                excerpt_end = excerptCache["end"]
                excerpt_article_question = article_question
                excerpt_article_zhihu_link = article_zhihu_link
                excerpt_article_author = article_author
                excerpt_article_favorites_count = article_favorites_count
                excerpt_article_last_update = article_last_update
                excerpt_article_links = article_links
                excerpt_article_title = article_title
                excerpt_article_tags = article_tags
                excerpt_article_read_count = article_read_count
                excerpt_article_content = article_content
                # 对相关文本字段进行向量化
                excerpt_vector = embeddings.embed_query(excerpt_quote)

                # 准备要插入的单条Excerpt数据字典
                excerpt_data_to_insert = {
                    "id": excerpt_rel_id,
                    "vector": excerpt_vector, 
                    "article_id": excerpt_article_id, 
                    "quote": excerpt_quote, 
                    "start_index": excerpt_start_index,
                    "create_time": excerpt_create_time,
                    "start": excerpt_start,
                    "excerpt_number": excerpt_excerpt_number,
                    "end_index": excerpt_end_index,
                    "end": excerpt_end,
                    "article_question" : excerpt_article_question,
                    "article_zhihu_link" : excerpt_article_zhihu_link,
                    "article_author" : excerpt_article_author,
                    "article_favorites_count" : excerpt_article_favorites_count,
                    "article_last_update" : excerpt_article_last_update,
                    "article_links" : excerpt_article_links,
                    "article_title" : excerpt_article_title,
                    "article_tags" : excerpt_article_tags,
                    "article_read_count" : excerpt_article_read_count,
                    "article_content" : excerpt_article_content
                }
                excerpt_data_to_insert_list.append(excerpt_data_to_insert)
            if len(excerpt_data_to_insert_list)>0:
                # 先加载集合
                client_set_load("excerpt_collection")
                client.insert("excerpt_collection", excerpt_data_to_insert_list)
                client.refresh_load("excerpt_collection")
            
        return res


def batch_insert_data(collection_name):
        #读取json文件
        file_path = 'data/caritas_data.json'
        data_list = []
        with open(file_path, 'r',encoding='utf-8') as file:
            data = json.load(file)
            data_list = data['data']

        article_data_to_insert_list = []
        excerpt_data_to_insert_list =[]
        i = 0
        for json_data in data_list:
            i = i+1
            print("i:",i)
            # if(i==20):
            #     break
            # 提取article字段
            article_rel_id = json_data["id"]
            article_question = json_data["question"] if 'question' in json_data else ''
            article_zhihu_link = json_data["zhihu_link"]
            article_author = json_data["author"]
            article_favorites_count = json_data["favorites_count"]
            article_last_update = json_data["last_update"]
            article_links = json_data["links"] if 'links' in json_data else ''
            article_title = json_data["title"] if 'title' in json_data else ''
            article_tags = json_data["tags"] if 'tags' in json_data else ''
            article_read_count = json_data["read_count"]
            article_content = json_data["content"]
            # 对相关文本字段进行向量化
            article_vector_content = embeddings.embed_query(article_content)
            article_vector_title = embeddings.embed_query(article_title)
            article_vector_question = embeddings.embed_query(article_question)

            # 准备要插入的单条article数据字典
            article_data_to_insert = {
                "id": article_rel_id,
                "vector_content": article_vector_content, 
                "vector_title": article_vector_title, 
                "vector_question": article_vector_question, 
                "question": article_question,
                "zhihu_link": article_zhihu_link,
                "author": article_author,
                "favorites_count": article_favorites_count,
                "last_update": article_last_update,
                "links": article_links,
                "title": article_title,
                "tags": article_tags,
                "content": article_content,
                "read_count": article_read_count
            }
            article_data_to_insert_list.append(article_data_to_insert)

            if "excerptCacheList" in json_data:
                for excerptCache in json_data['excerptCacheList']:
                    # 提取Excerpt字段
                    excerpt_rel_id = excerptCache["id"]
                    excerpt_article_id = excerptCache["article_id"]
                    excerpt_quote = excerptCache["quote"]
                    excerpt_start_index = excerptCache["start_index"]
                    excerpt_create_time = excerptCache["create_time"]
                    excerpt_start = excerptCache["start"]
                    excerpt_excerpt_number = excerptCache["excerpt_number"]
                    excerpt_end_index = excerptCache["end_index"]
                    excerpt_end = excerptCache["end"]
                    excerpt_article_question = article_question
                    excerpt_article_zhihu_link = article_zhihu_link
                    excerpt_article_author = article_author
                    excerpt_article_favorites_count = article_favorites_count
                    excerpt_article_last_update = article_last_update
                    excerpt_article_links = article_links
                    excerpt_article_title = article_title
                    excerpt_article_tags = article_tags
                    excerpt_article_read_count = article_read_count
                    excerpt_article_content = article_content
                    # 对相关文本字段进行向量化
                    excerpt_vector = embeddings.embed_query(excerpt_quote)
                    # 准备要插入的单条Excerpt数据字典
                    excerpt_data_to_insert = {
                        "id": excerpt_rel_id,
                        "vector": excerpt_vector, 
                        "article_id": excerpt_article_id, 
                        "quote": excerpt_quote, 
                        "start_index": excerpt_start_index,
                        "create_time": excerpt_create_time,
                        "start": excerpt_start,
                        "excerpt_number": excerpt_excerpt_number,
                        "end_index": excerpt_end_index,
                        "end": excerpt_end,
                        "article_question" : excerpt_article_question,
                        "article_zhihu_link" : excerpt_article_zhihu_link,
                        "article_author" : excerpt_article_author,
                        "article_favorites_count" : excerpt_article_favorites_count,
                        "article_last_update" : excerpt_article_last_update,
                        "article_links" : excerpt_article_links,
                        "article_title" : excerpt_article_title,
                        "article_tags" : excerpt_article_tags,
                        "article_read_count" : excerpt_article_read_count,
                        "article_content" : excerpt_article_content
                    }
                    excerpt_data_to_insert_list.append(excerpt_data_to_insert)

        if len(article_data_to_insert_list)>0:
            # 先加载集合
            client_set_load(collection_name)
            for i in range(0, len(article_data_to_insert_list), batch_size):
                batch_data = article_data_to_insert_list[i:i + batch_size]
                client.insert(
                    collection_name=collection_name,
                    data=batch_data
                )
            client.refresh_load(collection_name)

        if len(excerpt_data_to_insert_list)>0:
            # 先加载集合
            client_set_load("excerpt_collection")
            for i in range(0, len(excerpt_data_to_insert_list), batch_size):
                batch_data = excerpt_data_to_insert_list[i:i + batch_size]
                client.insert(
                    collection_name="excerpt_collection", 
                    data=batch_data
                )
            client.refresh_load("excerpt_collection")    


def delete_data(collection_name, record_id):
        """
        从指定的Milvus集合中删除指定ID的记录

        :param collection: Milvus集合对象
        :param record_id: 要删除记录的ID
        :return: 无
        """
        # 先加载集合
        client_set_load(collection_name=collection_name)
        client.delete(collection_name=collection_name, ids=[record_id])
        client.refresh_load(collection_name=collection_name)

        client_set_load(collection_name="excerpt_collection")
        res = client.delete(collection_name="excerpt_collection", filter="article_id == " + str(record_id))
        client.refresh_load(collection_name="excerpt_collection")
        return res


def update_data(collection_name, updated_data):
        """
        更新指定的Milvus集合中指定ID的记录数据

        :param collection: Milvus集合对象
        :param record_id: 要更新记录的ID
        :param updated_data: 包含更新后数据的字典，键应与集合模式中的字段名对应
        :return: 无
        """
        # client.upsert(collection_name, [updated_data])
        delete_data(collection_name,updated_data['id'])
        res = insert_data(collection_name,updated_data)
        client.refresh_load(collection_name)
        return res

def query_article_data(collection_name, query_vector, top_k=5):
        """
        根据输入的查询文本，在指定的Milvus集合中查询与之最相似的记录

        :param collection: Milvus集合对象
        :param query_vector: 输入的查询向量
        :param top_k: 要返回的最相似记录的数量，默认为5
        :return: 返回最相似记录的相关信息列表，每个元素为一个字典，包含记录的各个字段信息
        """
    
        # 创建AnnSearchRequest实例1
        request_1 = AnnSearchRequest(
            data=[query_vector],
            anns_field="vector_question",
            param={
                "metric_type": "COSINE", 
                "params": {"nprobe": 10}
            },
            limit=top_k
        )

        # 创建AnnSearchRequest实例2
        request_2 = AnnSearchRequest(
            data=[query_vector],
            anns_field="vector_title",
            param={
                "metric_type": "COSINE", 
                "params": {"nprobe": 10}
            },
            limit=top_k
        )

        # 创建AnnSearchRequest实例3
        # request_3 = AnnSearchRequest(
        #     data=[query_vector],
        #     anns_field="vector_content",
        #     param={
        #         "metric_type": "COSINE", 
        #         "params": {"nprobe": 10}
        #     },
        #     limit=top_k
        # )


        # 组装AnnSearchRequest实例列表
        reqs = [request_1, request_2]

        #配置重排策略
        rerank = WeightedRanker(0.6, 0.4) 

        #获取集合
        client_set_load(collection_name)
        collection = Collection(collection_name)

        #设置输出字段
        output_fields = ["id","question","zhihu_link","author","favorites_count","last_update","links","title","tags","content","read_count"]

        #执行混合查询
        res = collection.hybrid_search(
                reqs = reqs, # List of AnnSearchRequests created in step 1
                rerank = rerank, # Reranking strategy specified in step 2
                limit = top_k*2, # Number of final search results to return
                output_fields = output_fields
                )

        for hits in res:
            # print("hitType:",type(hit))
            # print("hit:",hit)
            hits_list = []
            for hit in hits:
                # print("hit_strType",type(hit))
                # print("hit_id_type",type(hit.id))
                # print("hit_str",hit.id)
                # print("hit_distance_type",type(hit.distance))
                # print("hit_distance",hit.distance)
                # print("hit_entity_type",type(hit.entity))
                # print("hit_entity",hit.entity.id)
                # print("hit.entity:",hit.entity)
                hit_dict = {
                        "id": hit.id,
                        "distance": hit.distance,
                        "entity": hit.entity.__dict__['fields'] ,# 将Hit转换为实际的字典
                        "model": "article"
                        }
                hits_list.append(hit_dict)

        result = json.dumps(hits_list, indent=4)
        result_obj = json.loads(result)
        return result_obj

def query_excerpt_data(collection_name, query_vector, top_k=5):
        # 设置查询参数（Milvus v2.4.x方式）
        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}, # nprobe是查询时的一个参数，可根据实际情况调整
        }
        output_fields = ["id", "article_id", "quote", "start_index", "create_time", "start", "excerpt_number", "end_index", "end", "article_question", "article_zhihu_link", "article_author", "article_favorites_count", "article_last_update", "article_links", "article_title", "article_tags", "article_read_count", "article_content"]

        client_set_load(collection_name)
        # 执行查询操作（Milvus v2.4.x方式）
        res = client.search(collection_name=collection_name, data=[query_vector], output_fields=output_fields,search_params=search_params, limit=top_k)
        re_list = []
        for re in res:
            for data in re:
                data['model'] = "excerpt"
                re_list.append(data)
        return re_list
