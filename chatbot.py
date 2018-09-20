#!/usr/bin/env python
# -*- coding: utf-8 -*-
__author__ = "chenzx"

from mednlp.utils.utils import pretty_print, load_json, match_patterns, strip_all_punctuations
from mednlp.dao.ai_service_dao import ai_services, transform_area, query_analyzer, transform_qa, greeting_service
import global_conf
import re
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


class TfidfSimilarity(object):
    def __init__(self, corpus_path):
        with open(corpus_path) as f:
            self.corpus = [strip_all_punctuations(s.strip()) for s in f.readlines()]
        self.tokenizer = jieba.cut
        self.vectorizer = CountVectorizer(tokenizer=self.tokenizer)
        self.transformer = TfidfTransformer()
        self.tfidf = self.transformer.fit_transform(self.vectorizer.fit_transform(self.corpus))
        self.word = self.vectorizer.get_feature_names()
        self.weight = self.tfidf.toarray()

    def jieba_cutall(self, q):
        return jieba.cut(q, cut_all=True)

    def verbose_corpus(self, n=10):
        for i in range(n):
            print u"\n"
            print self.corpus[i]
            for j in range(len(self.word)):
                if self.weight[i][j] > 0.00:
                    print self.word[j], self.weight[i][j]

    def verbose_q(self, q):
        q_tfidf_weight = self.transformer.transform(
            self.vectorizer.transform([strip_all_punctuations(q.strip())])).toarray()
        q_tokens = "|".join(self.tokenizer(q.strip())).split("|")
        print q_tokens
        print q_tfidf_weight
        print self.word
        for w, t in zip(q_tfidf_weight[0], self.word):
            if w > 0.0:
                print t, w

    def cosine(self, q, match_corpus=[], verbose=False):
        """
        计算余弦相似，加了一点规则
        :param q: 问句
        :param match_corpus: 搜索得到的相似自己
        :param verbose: 是否打印一些中间过程，用于调试
        :return: 相似问句或者None
        """

        q_tfidf_weight = self.transformer.transform(
            self.vectorizer.transform([strip_all_punctuations(q.strip())])).toarray()

        if match_corpus:
            match_corpus_strip = [strip_all_punctuations(s.strip()) for s in match_corpus]
            corpus_tfidf_weight = self.transformer.transform(
                self.vectorizer.transform(match_corpus_strip)).toarray()
            sim_vec = np.dot(corpus_tfidf_weight, np.transpose(q_tfidf_weight))
            sim_arg = np.argmax(sim_vec)
            best_q = match_corpus[sim_arg]
            best_score = np.max(sim_vec)
        else:
            sim_vec = np.dot(self.weight, np.transpose(q_tfidf_weight))
            sim_arg = np.argmax(sim_vec)
            best_q = self.corpus[sim_arg]
            best_score = np.max(sim_vec)

        if verbose:
            print "best_score:", best_score
            self.verbose_q(q)
            print best_q
            self.verbose_q(best_q)

        # 词数小于4的，所有词都必须在相似问句中出现
        q_tokens = "|".join(self.tokenizer(q.strip())).split("|")
        best_q_tokens = "|".join(self.tokenizer(best_q.strip())).split("|")
        if len(q_tokens) < 4:
            if not set(q_tokens).issubset(set(best_q_tokens)):
                best_score = 0

        if best_score > 0.4:
            return best_q, sim_arg
        else:
            return None, None

    def get_search_results(self, q):
        answers, err_msg = greeting_service(q)
        if err_msg:  # 无返回数据也会报错
            pretty_print(err_msg)
            return [], []
        return [strip_all_punctuations(s.get('ask')) for s in answers], [s.get('answer') for s in answers]

    def best_answer(self, q):
        search_ask, search_answer = self.get_search_results(q)
        if search_ask:
            best_q, sim_arg = self.cosine(q, search_ask)
            if best_q:
                return search_answer[sim_arg]


class Intention(object):
    """
    功能：
    通过正则文件判断句式；
    关键字句式优先query analyzer，entity_extract来判断,支持多关键词；
    通过句子中包含的实体情况，判断该句式是否成立，不成立的改句式为other
    """
    sen_type_keyword = 'keyword'
    sen_type_multi_keywords = 'multiKeywords'
    sen_type_other = 'other'
    sen_type_corpus_greeting = 'corpusGreeting'
    sen_type_greeting = 'greeting'
    sen_type_guide = 'guide'
    sen_type_customer_service = 'customerService'
    sen_type_content = 'content'
    regex = load_json(global_conf.model_dir + 'semantic_regex.json')
    field_mapping = {'std_department': 'department'}
    sentence_required_field = load_json(global_conf.model_dir + 'sentence_require_field.json')
    extract_entity_field = { }  # 仅去掉normal
    keyword_field_with_prior = []
    stop_words = [u',', u'，', u'、', u' ', u'？', u'?', u'.']

    err_msgs = dict()
    with open(global_conf.model_dir + 'greeting_regex.txt') as f:
        greeting_set = [s.strip().decode() for s in f.readlines()]

    with open(global_conf.model_dir + 'customerService_regex.txt') as f:
        customerService_set = [s.strip().decode() for s in f.readlines()]

    def _get_entities(self, query, fields=set(), need_id=True, stop=0):
        """
        从entity_extract得到实体后，将实体type统一为本类要求的形式，并根据条件过滤实体
        :param query: string
        :param fields: 留下的实体类型
        :param need_id: 是否需要id
        :param stop: 是否开启停用词
        :return: [实体,]
        """

        param = {'q': query.encode('utf8'),
                 'stopword': stop}
        entities, err_msg = ai_services(param, 'entity_extract', 'post')
        Intention.err_msgs.update(err_msg)
        if not entities:
            return []

        entities = transform_area(entities)
        refined_entities = []

        for seg in entities:
            if seg.get('type') in Intention.field_mapping.keys():
                seg['type'] = Intention.field_mapping[seg['type']]

            if not fields:
                if not need_id:
                    refined_entities.append(seg)
                if need_id and seg.get('entity_id'):
                    refined_entities.append(seg)

            if fields and seg.get('type') in fields:
                if not need_id:
                    refined_entities.append(seg)
                if need_id and seg.get('entity_id'):
                    refined_entities.append(seg)
        return refined_entities

    def check_greeting_intention(self, query):
        return match_patterns(unicode(strip_all_punctuations(query)), Intention.greeting_set)

    def check_customerService_intention(self, query):
        return match_patterns(unicode(strip_all_punctuations(query)), Intention.customerService_set)

    def check_guide_intention(self, query):
        """
        check the key word
        """
        words = u"(更新|预约)"
        if not re.search(words, query):
            return Intention.sen_type_guide
        else:
            return Intention.sen_type_other

    def _check_keyword_pipeline(self, segs):
        def drop_stop(seg):
            return seg.get('entity_name') not in Intention.stop_words

        def drop_unwanted_field(seg):
            return seg.get('type') in Intention.keyword_field_with_prior

        def center_word_index(segs):
            rank = [Intention.keyword_field_with_prior.index(seg.get('type')) for seg in segs]
            return rank.indexZZ(min(rank))

        def all_with_id(segs):
            segs_id = [seg.get('entity_id', 'null') for seg in segs]
            return 'null' not in segs_id

        keeped_segs = filter(drop_stop, segs)
        if keeped_segs and all_with_id(keeped_segs):
            target_segs = filter(drop_unwanted_field, keeped_segs)
            intention_and_entities = {}
            intention_and_entities['entities'] = target_segs

            if len(target_segs) == 1:
                intention_and_entities['intention'] = Intention.sen_type_keyword
                intention_and_entities['intentionDetails'] = target_segs[0].get('type')
                return intention_and_entities

            elif len(target_segs) > 1:
                intention_and_entities['intention'] = Intention.sen_type_multi_keywords
                intention_and_entities['intentionDetails'] = target_segs[center_word_index(target_segs)].get('type')
                return intention_and_entities

    def check_keyword_intentions(self, query):
        """
        分词，去停用词，判断是否全是id；id为1个，keyword;id多余1个，keywords,并给出中心词
        优先qa，不行就extract_entity
        :param query:
        :return:
        """
        analyzed_query, err_msg = query_analyzer(query.encode('utf-8'))
        Intention.err_msgs.update(err_msg)
        obj = self._check_keyword_pipeline(transform_qa(analyzed_query))
        if obj:
            return obj

        entities = self._get_entities(query, set(), need_id=False, stop=0)
        obj = self._check_keyword_pipeline(entities)
        if obj:
            return obj

    def check_intention_by_regex(self, query):
        """
        1、通过正则得出正则相关的句式
        2、其他句式丢到'other'
        :return:句式名字符串
        """
        query = unicode(query)
        for ints_to_pats in Intention.regex:
            if match_patterns(query, ints_to_pats.values()[0]):
                return ints_to_pats.keys()[0]
        return Intention.sen_type_other

    def _switch_entity_type(self, wanted, entities):
        """
        从实体的所有可能的类型当中，获取想要的类型
        :param wanted: 想要的
        :param entities: 实体集
        :return: 实体集
        """
        if not wanted:
            return entities

        for seg in entities:
            if wanted in seg.get('type_all'):
                seg['type'] = wanted
        return entities

    def _count_entities(self, entities):
        """
        统计实体类型出现次数
        :return:
        """
        count = {}
        for seg in entities:
            _type = seg.get('type')
            count.setdefault(_type, 0)
            count[_type] += 1
        return count

    def check_intention_and_entities(self, sen_type, entities):
        """
        1、检查实体类型，若句式下目标实体缺失，则判断备用类型有无目标类型，并返回替换原类型后的实体集，
        2、之后，判断句式是否满实体类型约束，否则转换句式到other
        :param sen_type: 句式
        :param entities: 实体集
        :return: 矫正后的句式和实体集
        """
        require = Intention.sentence_required_field.get(sen_type)
        if not require:
            return sen_type, entities

        # 根据wanted_types配置，当目标类型不存在时，启用备选字段
        count = self._count_entities(entities)
        wanted_list = []
        if require.get('wanted_types'):
            wanted_list = require.pop('wanted_types')
            for wanted in wanted_list:
                if not count.get('wanted'):
                    entities = self._switch_entity_type(wanted, entities)
            count = self._count_entities(entities)

        # 判断实体类型计数是否符合配置
        break_flag = False
        for k, v in require.items():
            if k == 'OR':
                cnt = 0
                for kk, kv in v.items():
                    if count.get(kk) == kv:
                        cnt += 1
                if cnt == 0:
                    break_flag = True
                    break
            else:
                if count.get(k) != v:
                    break_flag = True
                    break
        if wanted_list:
            require.update({'wanted_types': wanted_list})

        if break_flag:
            return Intention.sen_type_other, entities
        return sen_type, entities

    def get_intention_and_entities(self, query):
        """
        类的入口，得到query包含的意图及实体
        :param query: 非空字符串
        :return:{}
        """

        query = unicode(query)

        if len(query) < 100:
            keyword_info = self.check_keyword_intentions(query)
            if keyword_info:
                return keyword_info

        info = {}
        regex_sen_type = self.check_intention_by_regex(query)
        refined_entities = self._get_entities(query, fields=Intention.extract_entity_field, need_id=False, stop=0)
        intention, entities = self.check_intention_and_entities(regex_sen_type, refined_entities)

        if entities and (intention == Intention.sen_type_other):
            intention = Intention.sen_type_content

        if intention == Intention.sen_type_other:
            corpus_greeting_answer = TfidfSimilarity(
                global_conf.train_data_path + '/medical_robot/hanxuan_corpus.txt').best_answer(query)
            if corpus_greeting_answer:
                return {'intention': Intention.sen_type_corpus_greeting,
                        'answer': corpus_greeting_answer}
            if self.check_customerService_intention(query):
                print self.check_customerService_intention(query)
                return {'intention': Intention.sen_type_customer_service}
            if self.check_greeting_intention(query):
                return {'intention': Intention.sen_type_greeting}
            intention = self.check_guide_intention(query)

        info['intention'] = intention
        if entities:
            info['entities'] = entities
        return info


class DialogueAnalysis(object):
    """
    功能：对话服务接口
    """
    answer_required_field = load_json(global_conf.model_dir + 'answer_require_field.json')
    search_service_restrict_field = {}
    department_intentions = {}
    one_round_intentions = {}

    def __init__(self):
        self.intention = Intention()

    def _filter_output_entities(self, checkout_entities):
        """
        输出实体的过滤。
        相关的逻辑：
        :param checkout_entities: 输入的实体
        :return: 输出的实体
        """
        if not checkout_entities:
            return {}
        keys = checkout_entities.keys()
        if 'departmentName' in keys and 'hospital_departmentName' not in keys:
            return checkout_entities

        if 'departmentName' in keys and 'hospital_departmentName' in keys:
            for k in ['hospital_departmentName', 'hospital_departmentId']:
                if checkout_entities.get(k):
                    checkout_entities.pop(k)
            return checkout_entities

        if 'departmentName' not in keys and 'hospital_departmentName' in keys:
            checkout_entities['departmentName'] = checkout_entities.pop('hospital_departmentName')
            if checkout_entities.get('hospital_departmentId'):
                checkout_entities.pop('hospital_departmentId')
            return checkout_entities

        return checkout_entities

    def checkout_entities(self, entities, fields):
        """
        按接口的要求输出实体
        :param entities: query中得到的实体
        :param fields: 输出实体的域
        :return: {doctorId:,doctorName:,....}
        """
        if not entities:
            return {}
        checkout_entities = {}
        for seg in entities:
            _type = seg.get('type')
            if _type in fields:
                if seg.get('entity_id_all'):
                    if _type + 'Id' in checkout_entities.keys():
                        checkout_entities[_type + 'Id'].extend(seg.get('entity_id_all'))
                    else:
                        checkout_entities.update([(_type + 'Id', seg.get('entity_id_all'))])
                    checkout_entities[_type + 'Id'] = list(set(checkout_entities.get(_type + 'Id')))

                if seg.get('entity_name'):
                    if _type + 'Name' in checkout_entities.keys():
                        checkout_entities[_type + 'Name'].append(seg.get('entity_name'))
                    else:
                        checkout_entities.update([(_type + 'Name', [seg.get('entity_name')])])
                    checkout_entities[_type + 'Name'] = list(set(checkout_entities.get(_type + 'Name')))

        if checkout_entities.get('nationName'):
            for k in {'cityId', 'cityName', 'provinceId', 'provinceName', 'nationId'}:
                if checkout_entities.get(k):
                    checkout_entities.pop(k)
        return self._filter_output_entities(checkout_entities)

    def _replace_area(self, target_fields):
        """
        将target_fields中的'province', 'city'替换为 area
        :param target_fields: 替换后的集合
        :return: 替换后的集合
        """
        area_in_fields = {'province', 'city'} & target_fields
        if area_in_fields:
            target_fields -= area_in_fields
            target_fields.add('area')
        return target_fields

    def _checkout_area(self, _input, fl):
        if not (fl.get('cityId') or fl.get('provinceId') or fl.get('nationName')):
            if _input.get('city'):
                fl['cityId'] = _input.get('city').split(',')
            if _input.get('province'):
                fl['provinceId'] = _input.get('province').split(',')
        return fl

    def need_conditions(self, ints_ents, _input):
        """
        根据用户query和通过交互输入的历史实体，判断缺失字段，并返回交互输入的历史实体
        :param ints_ents:意图和实体
        :param _input:{}
        :return:{}.
        """

        required_fields_all = DialogueAnalysis.answer_required_field.get(ints_ents.get('intention'), {})
        required_fields = set(required_fields_all.keys())

        _fields_in_query = {seg.get('type') for seg in ints_ents.get('entities', [])}
        fields_in_query = self._replace_area(_fields_in_query)
        query_lack_fields = required_fields - fields_in_query

        if 'area' in query_lack_fields:
            query_lack_field_details = query_lack_fields | {'city', 'province'}
            query_lack_field_details.remove('area')
        else:
            query_lack_field_details = query_lack_fields

        _user_feedback_fields = set()
        user_feedback_entities = {}
        for field in query_lack_field_details:
            if field in {'city', 'province'}:
                if _input.get(field):
                    if _input.get(field):
                        _user_feedback_fields.add(field)
                        user_feedback_entities[field + 'Id'] = _input.get(field).split(',')
            else:
                if _input.get(field) or _input.get(field + 'Name'):
                    _user_feedback_fields.add(field)
                    if _input.get(field):
                        user_feedback_entities[field + 'Id'] = _input.get(field).split(',')
                    if _input.get(field + 'Name'):
                        user_feedback_entities[field + 'Name'] = _input.get(field + 'Name').split(',')

        user_feedback_fields = self._replace_area(_user_feedback_fields)

        matched_flag = True if not query_lack_fields or query_lack_fields == user_feedback_fields else False
        return query_lack_fields, user_feedback_entities, matched_flag

    def interact(self, query_body, debug=False):
        """
        交互输出
        :param query_body:用户询问的obj
        :return: {}
        """

        _input = query_body.get('input')
        ints_ents = self.intention.get_intention_and_entities(_input.get('q'))
        intention = ints_ents.get('intention')
        entities = ints_ents.get('entities')

        fl = {'isEnd': 0,
              'intention': intention}

        if ints_ents.get('intentionDetails'):
            fl['intentionDetails'] = [ints_ents.get('intentionDetails')]

        if debug:
            fl['debug'] = ints_ents
            fl['debug'].update({'regex_type': self.intention.check_intention_by_regex(_input.get('q'))})

        if self.intention.err_msgs:
            fl['err_msgs'] = self.intention.err_msgs

        if intention == self.intention.sen_type_corpus_greeting:
            fl['intention'] = self.intention.sen_type_greeting
            fl['answer'] = ints_ents.get('answer')
            fl['isEnd'] = 1
            return fl

        if intention == self.intention.sen_type_greeting:
            fl['answer'] = '你好，请问有什么可以帮助您？'
            fl['isEnd'] = 1
            return fl

        if intention == self.intention.sen_type_customer_service:
            fl['answer'] = '不懂您在说什么，请问需要呼叫人工客服吗？'
            fl['isEnd'] = 1
            return fl

        if intention == self.intention.sen_type_guide:
            # 临时支撑，包一层
            fl['intention'] = self.intention.sen_type_greeting
            fl['answer'] = '你好，目前仅支持**相关问题，请问有什么可以帮您吗？'
            fl['isEnd'] = 1
            return fl

        if intention == self.intention.sen_type_other:
            fl['isHelp'] = 1
            fl['isEnd'] = 1
            return fl

        if intention == self.intention.sen_type_content:
            fl['isEnd'] = 1
            # 临时支持
            # fl.update(self.checkout_entities(entities, self.intention.keyword_field_with_prior))
            fl['intention'] = self.intention.sen_type_keyword
            fl['treatmentName'] = [_input.get('q')]
            fl['intentionDetails'] = ['treatment']
            return fl

        if intention in [self.intention.sen_type_keyword, self.intention.sen_type_multi_keywords]:
            fl['isEnd'] = 1
            fl['intention'] = self.intention.sen_type_keyword
            fl.update(self.checkout_entities(entities, self.intention.keyword_field_with_prior))
            return self._checkout_area(_input, fl)

        fl.update(self.checkout_entities(entities, DialogueAnalysis.search_service_restrict_field))
        if intention in DialogueAnalysis.one_round_intentions:
            fl['isEnd'] = 1
            return self._checkout_area(_input, fl)

        if intention in DialogueAnalysis.department_intentions:
            param = {'rows': 1}
            param['source'] = query_body.get('source')
            param['q'] = _input.get('q').encode('utf8')
            if _input.get('symptomName'):
                param['symptom'] = _input.get('symptomName').encode('utf8')
            if _input.get('sex'):
                param['sex'] = _input.get('sex')
            if _input.get('age'):
                param['age'] = _input.get('age')
            interrupt = True if _input.get('isEnd') else False
            dept_fl = self.department_interact(param, fl, interrupt)
            fl.update(dept_fl)
            else:
                fl.pop('department_updated')
            if self.intention.err_msgs and dept_fl.get('err_msgs'):
                fl.get('err_msgs').update(self.intention.err_msgs)
            return self._checkout_area(_input, fl)

        query_lack_fields, user_feedback_entities, matched_flag = self.need_conditions(ints_ents, _input)

        if _input.get('isEnd'):
            fl_interrupt = dict()
            fl_interrupt['intention'] = self.intention.sen_type_other
            fl_interrupt['isEnd'] = 1
            fl_interrupt['isHelp'] = 1
            return fl_interrupt

        if matched_flag:
            fl['isEnd'] = 1
            if query_lack_fields:
                fl.update(user_feedback_entities)
        else:
            if ints_ents.get('entities'):
                for i in ints_ents.get('entities').values:
                    if i[u'doctor']:
                        fl['isEnd'] = 1

        return fl
